# Multi-Agent Business Intelligence Platform for Databricks
# Enterprise-grade natural language to insights pipeline

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple
from uuid import uuid4

import pandas as pd
import sqlparse
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql as databricks_sql
from pydantic import BaseModel, Field


# =============================================================================
# CONFIGURATION & MODELS
# =============================================================================

class RequestType(Enum):
    QUERY = "query"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    ALERT_SETUP = "alert_setup"
    ALERT_CHECK = "alert_check"


class AgentType(Enum):
    ORCHESTRATOR = "orchestrator"
    NL_SQL = "nl_sql"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    ALERTING = "alerting"


class SecurityLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class UserContext:
    """User context for security and personalization"""
    user_id: str
    email: str
    department: str
    roles: List[str]
    security_clearance: SecurityLevel
    accessible_schemas: List[str]
    accessible_tables: List[str]


@dataclass
class QueryRequest:
    """Unified request model for all agent interactions"""
    request_id: str
    user_context: UserContext
    request_type: RequestType
    natural_language_query: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QueryResult:
    """Standardized result format across agents"""
    request_id: str
    agent_type: AgentType
    success: bool
    data: Optional[pd.DataFrame] = None
    sql_query: Optional[str] = None
    explanation: Optional[str] = None
    visualization_url: Optional[str] = None
    error_message: Optional[str] = None
    execution_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KPIDefinition:
    """KPI monitoring configuration"""
    kpi_id: str
    name: str
    sql_query: str
    threshold_value: float
    comparison_operator: str  # >, <, >=, <=, ==, !=
    alert_recipients: List[str]
    check_frequency: str  # cron expression
    enabled: bool = True
    created_by: str = ""
    last_value: Optional[float] = None
    last_checked: Optional[datetime] = None


# =============================================================================
# SECURITY & GOVERNANCE
# =============================================================================

class SecurityManager:
    """Handles access control, PII masking, and audit logging"""
    
    def __init__(self, unity_catalog_client):
        self.unity_catalog = unity_catalog_client
        self.audit_log = []
        
        # PII patterns for masking
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        }
    
    def validate_access(self, user_context: UserContext, schema: str, table: str) -> bool:
        """Validate user access to specific tables"""
        try:
            # Check schema access
            if schema not in user_context.accessible_schemas:
                self.log_security_event(
                    user_context.user_id,
                    "SCHEMA_ACCESS_DENIED",
                    f"User attempted to access schema: {schema}"
                )
                return False
            
            # Check table access
            full_table = f"{schema}.{table}"
            if full_table not in user_context.accessible_tables:
                self.log_security_event(
                    user_context.user_id,
                    "TABLE_ACCESS_DENIED",
                    f"User attempted to access table: {full_table}"
                )
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Access validation error: {e}")
            return False
    
    def mask_pii(self, data: pd.DataFrame) -> pd.DataFrame:
        """Mask PII in query results"""
        masked_data = data.copy()
        
        for column in masked_data.columns:
            if masked_data[column].dtype == 'object':
                for pii_type, pattern in self.pii_patterns.items():
                    masked_data[column] = masked_data[column].astype(str).str.replace(
                        pattern, f"[MASKED_{pii_type.upper()}]", regex=True
                    )
        
        return masked_data
    
    def validate_sql_safety(self, sql_query: str) -> Tuple[bool, Optional[str]]:
        """Validate SQL for safety (no DDL/DML, injection protection)"""
        try:
            # Parse SQL
            parsed = sqlparse.parse(sql_query.upper())
            
            # Check for dangerous operations
            dangerous_keywords = [
                'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE',
                'TRUNCATE', 'EXEC', 'EXECUTE', 'DECLARE', 'MERGE'
            ]
            
            for statement in parsed:
                tokens = [token.ttype for token in statement.flatten()]
                sql_text = str(statement).upper()
                
                for keyword in dangerous_keywords:
                    if keyword in sql_text:
                        return False, f"Dangerous SQL operation detected: {keyword}"
            
            return True, None
            
        except Exception as e:
            return False, f"SQL parsing error: {e}"
    
    def log_security_event(self, user_id: str, event_type: str, details: str):
        """Log security events for audit"""
        self.audit_log.append({
            'timestamp': datetime.now(),
            'user_id': user_id,
            'event_type': event_type,
            'details': details
        })
        logging.warning(f"Security Event - {event_type}: {details} (User: {user_id})")


# =============================================================================
# BASE AGENT FRAMEWORK
# =============================================================================

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, agent_type: AgentType, workspace_client: WorkspaceClient):
        self.agent_type = agent_type
        self.workspace_client = workspace_client
        self.logger = logging.getLogger(f"{agent_type.value}_agent")
        
    @abstractmethod
    async def process(self, request: QueryRequest) -> QueryResult:
        """Process a request and return results"""
        pass
    
    def _create_result(self, request: QueryRequest, success: bool, **kwargs) -> QueryResult:
        """Helper to create standardized results"""
        return QueryResult(
            request_id=request.request_id,
            agent_type=self.agent_type,
            success=success,
            **kwargs
        )


# =============================================================================
# ORCHESTRATOR AGENT
# =============================================================================

class OrchestratorAgent(BaseAgent):
    """Central coordinator for all agent interactions"""
    
    def __init__(self, workspace_client: WorkspaceClient, security_manager: SecurityManager):
        super().__init__(AgentType.ORCHESTRATOR, workspace_client)
        self.security_manager = security_manager
        self.agents = {}
        self.request_state = {}
        
    def register_agent(self, agent_type: AgentType, agent: BaseAgent):
        """Register an agent with the orchestrator"""
        self.agents[agent_type] = agent
        self.logger.info(f"Registered {agent_type.value} agent")
    
    async def process(self, request: QueryRequest) -> QueryResult:
        """Main orchestration logic"""
        start_time = time.time()
        
        try:
            # Store request state
            self.request_state[request.request_id] = {
                'status': 'processing',
                'steps': [],
                'start_time': start_time
            }
            
            # Route based on request type
            if request.request_type == RequestType.QUERY:
                result = await self._handle_query_flow(request)
            elif request.request_type == RequestType.ANALYSIS:
                result = await self._handle_analysis_flow(request)
            elif request.request_type == RequestType.VISUALIZATION:
                result = await self._handle_visualization_flow(request)
            elif request.request_type == RequestType.ALERT_SETUP:
                result = await self._handle_alert_setup_flow(request)
            elif request.request_type == RequestType.ALERT_CHECK:
                result = await self._handle_alert_check_flow(request)
            else:
                result = self._create_result(
                    request, False,
                    error_message=f"Unsupported request type: {request.request_type}"
                )
            
            # Update execution time
            result.execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Update request state
            self.request_state[request.request_id]['status'] = 'completed'
            self.request_state[request.request_id]['result'] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Orchestration error: {e}")
            return self._create_result(
                request, False,
                error_message=f"Orchestration failed: {str(e)}",
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def _handle_query_flow(self, request: QueryRequest) -> QueryResult:
        """Handle natural language query flow"""
        # Step 1: NL→SQL conversion and execution
        nl_sql_agent = self.agents[AgentType.NL_SQL]
        sql_result = await nl_sql_agent.process(request)
        
        if not sql_result.success:
            return sql_result
        
        # Step 2: Optional analysis if requested
        if request.metadata.get('include_analysis', False):
            analysis_request = QueryRequest(
                request_id=f"{request.request_id}_analysis",
                user_context=request.user_context,
                request_type=RequestType.ANALYSIS,
                natural_language_query=f"Analyze this data: {request.natural_language_query}",
                metadata={'source_data': sql_result.data}
            )
            
            analysis_agent = self.agents[AgentType.ANALYSIS]
            analysis_result = await analysis_agent.process(analysis_request)
            
            # Merge results
            sql_result.explanation = f"{sql_result.explanation}\n\nAnalysis: {analysis_result.explanation}"
        
        return sql_result
    
    async def _handle_analysis_flow(self, request: QueryRequest) -> QueryResult:
        """Handle analysis request flow"""
        analysis_agent = self.agents[AgentType.ANALYSIS]
        return await analysis_agent.process(request)
    
    async def _handle_visualization_flow(self, request: QueryRequest) -> QueryResult:
        """Handle visualization request flow"""
        viz_agent = self.agents[AgentType.VISUALIZATION]
        return await viz_agent.process(request)
    
    async def _handle_alert_setup_flow(self, request: QueryRequest) -> QueryResult:
        """Handle alert setup flow"""
        alert_agent = self.agents[AgentType.ALERTING]
        return await alert_agent.process(request)
    
    async def _handle_alert_check_flow(self, request: QueryRequest) -> QueryResult:
        """Handle alert checking flow"""
        alert_agent = self.agents[AgentType.ALERTING]
        return await alert_agent.process(request)


# =============================================================================
# NL→SQL AGENT
# =============================================================================

class NLToSQLAgent(BaseAgent):
    """Converts natural language to SQL and executes queries"""
    
    def __init__(self, workspace_client: WorkspaceClient, security_manager: SecurityManager):
        super().__init__(AgentType.NL_SQL, workspace_client)
        self.security_manager = security_manager
        self.schema_cache = {}
        self.query_cache = {}
        
    async def process(self, request: QueryRequest) -> QueryResult:
        """Convert NL to SQL, validate, execute, and return results"""
        start_time = time.time()
        
        try:
            # Step 1: Analyze natural language query
            query_intent = await self._analyze_query_intent(request.natural_language_query)
            
            # Step 2: Get relevant schema information
            schema_info = await self._get_schema_context(
                request.user_context, query_intent['tables']
            )
            
            # Step 3: Generate SQL
            sql_query = await self._generate_sql(
                request.natural_language_query, schema_info, query_intent
            )
            
            # Step 4: Validate SQL safety
            is_safe, safety_error = self.security_manager.validate_sql_safety(sql_query)
            if not is_safe:
                return self._create_result(
                    request, False,
                    error_message=f"SQL safety validation failed: {safety_error}",
                    sql_query=sql_query
                )
            
            # Step 5: Execute query
            data = await self._execute_sql(sql_query, request.user_context)
            
            # Step 6: Apply PII masking
            masked_data = self.security_manager.mask_pii(data)
            
            # Step 7: Generate explanation
            explanation = await self._generate_explanation(
                request.natural_language_query, sql_query, masked_data, query_intent
            )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return self._create_result(
                request, True,
                data=masked_data,
                sql_query=sql_query,
                explanation=explanation,
                execution_time_ms=execution_time,
                metadata={
                    'query_intent': query_intent,
                    'schema_tables_used': query_intent['tables'],
                    'row_count': len(masked_data)
                }
            )
            
        except Exception as e:
            self.logger.error(f"NL→SQL processing error: {e}")
            return self._create_result(
                request, False,
                error_message=f"Query processing failed: {str(e)}",
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def _analyze_query_intent(self, nl_query: str) -> Dict[str, Any]:
        """Analyze natural language query to understand intent"""
        # Simplified intent analysis - in production, use LLM
        query_lower = nl_query.lower()
        
        intent = {
            'type': 'select',
            'tables': [],
            'columns': [],
            'filters': [],
            'aggregations': [],
            'time_range': None,
            'metrics': []
        }
        
        # Extract common business entities
        business_entities = {
            'revenue': ['sales', 'finance'],
            'customers': ['customers', 'users'],
            'orders': ['orders', 'transactions'],
            'products': ['products', 'inventory'],
            'employees': ['hr', 'employees']
        }
        
        for entity, schemas in business_entities.items():
            if entity in query_lower:
                intent['tables'].extend([f"{schema}.{entity}" for schema in schemas])
        
        # Extract time-based filters
        time_keywords = ['q1', 'q2', 'q3', 'q4', 'quarter', 'month', 'year', 'week']
        for keyword in time_keywords:
            if keyword in query_lower:
                intent['time_range'] = keyword
                break
        
        # Extract aggregation intent
        agg_keywords = ['sum', 'count', 'average', 'max', 'min', 'total']
        for keyword in agg_keywords:
            if keyword in query_lower:
                intent['aggregations'].append(keyword)
        
        return intent
    
    async def _get_schema_context(self, user_context: UserContext, tables: List[str]) -> Dict[str, Any]:
        """Get schema information for relevant tables"""
        schema_context = {}
        
        try:
            # In production, query Unity Catalog for schema information
            # This is a simplified version
            for table in tables:
                if table in user_context.accessible_tables:
                    # Mock schema info - replace with actual Unity Catalog queries
                    schema_context[table] = {
                        'columns': self._get_table_columns(table),
                        'relationships': self._get_table_relationships(table),
                        'sample_data': self._get_sample_data(table)
                    }
        
        except Exception as e:
            self.logger.error(f"Schema context error: {e}")
        
        return schema_context
    
    def _get_table_columns(self, table: str) -> List[Dict[str, str]]:
        """Get column information for a table"""
        # Mock column data - replace with Unity Catalog queries
        column_mappings = {
            'sales.revenue': [
                {'name': 'revenue_id', 'type': 'int', 'description': 'Unique revenue record ID'},
                {'name': 'amount', 'type': 'decimal', 'description': 'Revenue amount in USD'},
                {'name': 'region', 'type': 'string', 'description': 'Geographic region'},
                {'name': 'quarter', 'type': 'string', 'description': 'Fiscal quarter (Q1, Q2, Q3, Q4)'},
                {'name': 'year', 'type': 'int', 'description': 'Fiscal year'},
                {'name': 'created_date', 'type': 'timestamp', 'description': 'Record creation date'}
            ],
            'customers.customers': [
                {'name': 'customer_id', 'type': 'int', 'description': 'Unique customer ID'},
                {'name': 'customer_name', 'type': 'string', 'description': 'Customer name'},
                {'name': 'email', 'type': 'string', 'description': 'Customer email'},
                {'name': 'signup_date', 'type': 'timestamp', 'description': 'Customer signup date'},
                {'name': 'churn_rate', 'type': 'decimal', 'description': 'Customer churn rate'}
            ]
        }
        
        return column_mappings.get(table, [])
    
    def _get_table_relationships(self, table: str) -> List[Dict[str, str]]:
        """Get relationship information for a table"""
        # Mock relationships - replace with Unity Catalog metadata
        return []
    
    def _get_sample_data(self, table: str) -> Dict[str, Any]:
        """Get sample data for context"""
        # Mock sample data - replace with actual sampling
        return {'sample_rows': 3, 'total_rows': 10000}
    
    async def _generate_sql(self, nl_query: str, schema_info: Dict[str, Any], intent: Dict[str, Any]) -> str:
        """Generate SQL from natural language"""
        # Simplified SQL generation - in production, use LLM with schema context
        
        # Example mappings for common queries
        if 'revenue' in nl_query.lower() and 'region' in nl_query.lower():
            if 'q2 2024' in nl_query.lower():
                return """
                SELECT 
                    region,
                    SUM(amount) as total_revenue
                FROM sales.revenue 
                WHERE quarter = 'Q2' AND year = 2024
                GROUP BY region
                ORDER BY total_revenue DESC
                """
        
        elif 'churn' in nl_query.lower() and 'week' in nl_query.lower():
            return """
                SELECT 
                    WEEK(signup_date) as week_number,
                    AVG(churn_rate) as avg_churn_rate
                FROM customers.customers 
                WHERE signup_date >= DATE_SUB(CURRENT_DATE(), 8)
                GROUP BY WEEK(signup_date)
                ORDER BY week_number DESC
                """
        
        elif 'subscriptions' in nl_query.lower() and 'month' in nl_query.lower():
            return """
                SELECT 
                    DATE_TRUNC('month', signup_date) as month,
                    COUNT(*) as new_subscriptions,
                    LAG(COUNT(*)) OVER (ORDER BY DATE_TRUNC('month', signup_date)) as prev_month,
                    (COUNT(*) - LAG(COUNT(*)) OVER (ORDER BY DATE_TRUNC('month', signup_date))) / 
                    LAG(COUNT(*)) OVER (ORDER BY DATE_TRUNC('month', signup_date)) * 100 as growth_rate
                FROM customers.customers
                WHERE signup_date >= DATE_SUB(CURRENT_DATE(), 365)
                GROUP BY DATE_TRUNC('month', signup_date)
                ORDER BY month DESC
                """
        
        # Fallback for unknown queries
        return "SELECT 'Query not recognized' as message"
    
    async def _execute_sql(self, sql_query: str, user_context: UserContext) -> pd.DataFrame:
        """Execute SQL query against Databricks"""
        try:
            # Check cache first
            cache_key = f"{hash(sql_query)}_{user_context.user_id}"
            if cache_key in self.query_cache:
                cache_entry = self.query_cache[cache_key]
                if datetime.now() - cache_entry['timestamp'] < timedelta(minutes=5):
                    self.logger.info("Returning cached result")
                    return cache_entry['data']
            
            # Execute query (mock data for demonstration)
            if 'revenue' in sql_query.lower():
                data = pd.DataFrame({
                    'region': ['North America', 'Europe', 'Asia Pacific', 'Latin America'],
                    'total_revenue': [2450000, 1890000, 1650000, 890000]
                })
            elif 'churn' in sql_query.lower():
                data = pd.DataFrame({
                    'week_number': [35, 34, 33, 32],
                    'avg_churn_rate': [0.025, 0.031, 0.028, 0.022]
                })
            elif 'subscriptions' in sql_query.lower():
                data = pd.DataFrame({
                    'month': ['2024-08', '2024-07', '2024-06', '2024-05'],
                    'new_subscriptions': [1250, 1180, 1090, 1150],
                    'prev_month': [1180, 1090, 1150, 1050],
                    'growth_rate': [5.93, 8.26, -5.22, 9.52]
                })
            else:
                data = pd.DataFrame({'message': ['Query not recognized']})
            
            # Cache result
            self.query_cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            return data
            
        except Exception as e:
            self.logger.error(f"SQL execution error: {e}")
            raise
    
    async def _generate_explanation(self, nl_query: str, sql_query: str, data: pd.DataFrame, intent: Dict[str, Any]) -> str:
        """Generate human-readable explanation of results"""
        
        explanation_parts = []
        
        # Query understanding
        explanation_parts.append(f"**Query Understanding:** {nl_query}")
        
        # SQL generated
        explanation_parts.append(f"**SQL Generated:** ```sql\n{sql_query}\n```")
        
        # Results summary
        if len(data) > 0:
            explanation_parts.append(f"**Results Summary:** Found {len(data)} records.")
            
            # Add specific insights based on data
            if 'revenue' in data.columns:
                total_revenue = data['total_revenue'].sum() if 'total_revenue' in data.columns else 0
                explanation_parts.append(f"Total revenue across all regions: ${total_revenue:,.2f}")
                
                if len(data) > 1:
                    top_region = data.iloc[0]
                    explanation_parts.append(f"Top performing region: {top_region.iloc[0]} with ${top_region.iloc[1]:,.2f}")
            
            elif 'churn_rate' in data.columns:
                avg_churn = data['avg_churn_rate'].mean() if 'avg_churn_rate' in data.columns else 0
                explanation_parts.append(f"Average churn rate: {avg_churn:.1%}")
                
                if len(data) > 1:
                    trend = "increasing" if data['avg_churn_rate'].iloc[0] > data['avg_churn_rate'].iloc[-1] else "decreasing"
                    explanation_parts.append(f"Churn trend: {trend} over the analyzed period")
            
            elif 'growth_rate' in data.columns:
                avg_growth = data['growth_rate'].mean() if 'growth_rate' in data.columns else 0
                explanation_parts.append(f"Average month-over-month growth: {avg_growth:.1f}%")
        else:
            explanation_parts.append("**Results Summary:** No data found matching the criteria.")
        
        return "\n\n".join(explanation_parts)


# =============================================================================
# ANALYSIS AGENT
# =============================================================================

class AnalysisAgent(BaseAgent):
    """Performs advanced data analysis and insights"""
    
    def __init__(self, workspace_client: WorkspaceClient):
        super().__init__(AgentType.ANALYSIS, workspace_client)
    
    async def process(self, request: QueryRequest) -> QueryResult:
        """Perform data analysis based on natural language request"""
        start_time = time.time()
        
        try:
            # Get source data
            if 'source_data' in request.metadata:
                data = request.metadata['source_data']
            else:
                # If no source data, trigger SQL agent first
                return self._create_result(
                    request, False,
                    error_message="Analysis requires source data. Please run a query first."
                )
            
            # Perform analysis
            analysis_results = await self._perform_analysis(
                data, request.natural_language_query
            )
            
            # Generate insights
            insights = await self._generate_insights(data, analysis_results)
            
            # Suggest follow-up questions
            follow_ups = await self._suggest_follow_ups(data, request.natural_language_query)
            
            explanation = f"""
**Analysis Results:**
{insights}

**Key Findings:**
{analysis_results.get('summary', 'Analysis completed')}

**Suggested Follow-up Questions:**
{chr(10).join(f"• {q}" for q in follow_ups)}
"""
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return self._create_result(
                request, True,
                data=data,
                explanation=explanation,
                execution_time_ms=execution_time,
                metadata={
                    'analysis_type': analysis_results.get('type', 'descriptive'),
                    'follow_up_questions': follow_ups
                }
            )
            
        except Exception as e:
            self.logger.error(f"Analysis error: {e}")
            return self._create_result(
                request, False,
                error_message=f"Analysis failed: {str(e)}",
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def _perform_analysis(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Perform statistical analysis on the data"""
        analysis = {
            'type': 'descriptive',
            'summary': '',
            'statistics': {},
            'trends': {},
            'anomalies': []
        }
        
        try:
            # Basic statistics
            numeric_columns = data.select_dtypes(include=['number']).columns
            if len(numeric_columns) > 0:
                analysis['statistics'] = {
                    'mean': data[numeric_columns].mean().to_dict(),
                    'median': data[numeric_columns].median().to_dict(),
                    'std': data[numeric_columns].std().to_dict()
                }
            
            # Trend analysis for time-series data
            if any(col in data.columns for col in ['month', 'quarter', 'week_number']):
                analysis['type'] = 'time_series'
                analysis['trends'] = self._analyze_trends(data)
            
            # Anomaly detection
            analysis['anomalies'] = self._detect_anomalies(data)
            
            # Generate summary
            if analysis['trends']:
                trend_direction = analysis['trends'].get('direction', 'stable')
                analysis['summary'] = f"Data shows a {trend_direction} trend with {len(analysis['anomalies'])} anomalies detected."
            else:
                analysis['summary'] = f"Descriptive analysis of {len(data)} records completed."
        
        except Exception as e:
            self.logger.error(f"Analysis computation error: {e}")
            analysis['summary'] = "Analysis completed with limited results due to data structure."
        
        return analysis
    
    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in time-series data"""
        trends = {}
        
        # Find numeric columns for trend analysis
        numeric_cols = data.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if len(data) >= 2:
                values = data[col].values
                if len(values) > 1:
                    # Simple trend calculation
                    slope = (values[-1] - values[0]) / len(values)
                    if slope > 0:
                        direction = "increasing"
                    elif slope < 0:
                        direction = "decreasing"
                    else:
                        direction = "stable"
                    
                    trends[col] = {
                        'direction': direction,
                        'slope': slope,
                        'change_percent': ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                    }
        
        # Overall trend direction
        if trends:
            directions = [t['direction'] for t in trends.values()]
            most_common = max(set(directions), key=directions.count)
            trends['direction'] = most_common
        
        return trends
    
    def _detect_anomalies(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect anomalies in numeric data"""
        anomalies = []
        
        numeric_cols = data.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            values = data[col].dropna()
            if len(values) > 3:
                # Simple z-score based anomaly detection
                mean_val = values.mean()
                std_val = values.std()
                
                if std_val > 0:
                    z_scores = abs((values - mean_val) / std_val)
                    anomaly_indices = z_scores[z_scores > 2].index.tolist()
                    
                    for idx in anomaly_indices:
                        anomalies.append({
                            'column': col,
                            'row_index': idx,
                            'value': data.loc[idx, col],
                            'z_score': z_scores[idx],
                            'severity': 'high' if z_scores[idx] > 3 else 'medium'
                        })
        
        return anomalies
    
    async def _generate_insights(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> str:
        """Generate human-readable insights from analysis"""
        insights = []
        
        # Statistical insights
        if 'statistics' in analysis_results:
            stats = analysis_results['statistics']
            for col, mean_val in stats.get('mean', {}).items():
                insights.append(f"Average {col}: {mean_val:.2f}")
        
        # Trend insights
        if 'trends' in analysis_results:
            trends = analysis_results['trends']
            for col, trend_info in trends.items():
                if isinstance(trend_info, dict) and 'direction' in trend_info:
                    change_pct = trend_info.get('change_percent', 0)
                    insights.append(f"{col} is {trend_info['direction']} ({change_pct:+.1f}%)")
        
        # Anomaly insights
        anomalies = analysis_results.get('anomalies', [])
        if anomalies:
            high_severity = [a for a in anomalies if a['severity'] == 'high']
            if high_severity:
                insights.append(f"⚠️ {len(high_severity)} high-severity anomalies detected")
        
        return "\n".join(insights) if insights else "No significant patterns detected."
    
    async def _suggest_follow_ups(self, data: pd.DataFrame, original_query: str) -> List[str]:
        """Suggest relevant follow-up questions"""
        suggestions = []
        
        # Based on data columns
        if 'region' in data.columns:
            suggestions.append("Which region shows the most growth potential?")
            suggestions.append("How do regional performance patterns compare year-over-year?")
        
        if 'churn_rate' in data.columns or 'churn' in original_query.lower():
            suggestions.append("What factors correlate with higher churn rates?")
            suggestions.append("Which customer segments have the lowest churn?")
        
        if 'revenue' in data.columns or 'revenue' in original_query.lower():
            suggestions.append("What are the revenue drivers in top-performing regions?")
            suggestions.append("How does revenue seasonality affect planning?")
        
        # Generic suggestions
        if len(suggestions) == 0:
            suggestions.extend([
                "What trends do you see in this data over time?",
                "Are there any correlations worth investigating?",
                "How does this compare to industry benchmarks?"
            ])
        
        return suggestions[:3]  # Limit to 3 suggestions


# =============================================================================
# VISUALIZATION AGENT
# =============================================================================

class VisualizationAgent(BaseAgent):
    """Handles visualization creation and publishing to BI tools"""
    
    def __init__(self, workspace_client: WorkspaceClient):
        super().__init__(AgentType.VISUALIZATION, workspace_client)
        self.tableau_config = self._load_tableau_config()
        self.powerbi_config = self._load_powerbi_config()
    
    async def process(self, request: QueryRequest) -> QueryResult:
        """Create and publish visualizations"""
        start_time = time.time()
        
        try:
            # Get data to visualize
            if 'source_data' not in request.metadata:
                return self._create_result(
                    request, False,
                    error_message="Visualization requires source data"
                )
            
            data = request.metadata['source_data']
            
            # Determine best visualization type
            viz_type = await self._recommend_visualization(data, request.natural_language_query)
            
            # Create visualization metadata
            viz_metadata = await self._create_visualization_metadata(data, viz_type)
            
            # Publish to BI tools
            tableau_url = await self._publish_to_tableau(data, viz_metadata)
            powerbi_url = await self._publish_to_powerbi(data, viz_metadata)
            
            explanation = f"""
**Visualization Created:**
- Type: {viz_type}
- Data Points: {len(data)} records
- Recommended Chart: {viz_metadata['chart_type']}

**Published To:**
- Tableau: {tableau_url or 'Publishing failed'}
- Power BI: {powerbi_url or 'Publishing failed'}

**Visualization Guidelines:**
{viz_metadata['guidelines']}
"""
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return self._create_result(
                request, True,
                data=data,
                explanation=explanation,
                visualization_url=tableau_url or powerbi_url,
                execution_time_ms=execution_time,
                metadata={
                    'visualization_type': viz_type,
                    'tableau_url': tableau_url,
                    'powerbi_url': powerbi_url,
                    'chart_recommendations': viz_metadata
                }
            )
            
        except Exception as e:
            self.logger.error(f"Visualization error: {e}")
            return self._create_result(
                request, False,
                error_message=f"Visualization failed: {str(e)}",
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
    
    def _load_tableau_config(self) -> Dict[str, Any]:
        """Load Tableau server configuration"""
        return {
            'server_url': 'https://your-tableau-server.com',
            'site_id': 'your-site',
            'project_name': 'BI_Analytics',
            'username': 'tableau_service_account',
            'password': 'tableau_password'  # Use secure credential management
        }
    
    def _load_powerbi_config(self) -> Dict[str, Any]:
        """Load Power BI configuration"""
        return {
            'tenant_id': 'your-tenant-id',
            'client_id': 'your-app-id',
            'client_secret': 'your-client-secret',  # Use secure credential management
            'workspace_id': 'your-workspace-id'
        }
    
    async def _recommend_visualization(self, data: pd.DataFrame, query: str) -> str:
        """Recommend the best visualization type for the data"""
        
        # Analyze data structure
        numeric_cols = data.select_dtypes(include=['number']).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        date_cols = data.select_dtypes(include=['datetime']).columns
        
        # Rules-based recommendation
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            return "time_series"
        elif len(categorical_cols) == 1 and len(numeric_cols) == 1:
            if len(data) <= 10:
                return "bar_chart"
            else:
                return "horizontal_bar"
        elif len(categorical_cols) >= 2 and len(numeric_cols) >= 1:
            return "grouped_bar"
        elif len(numeric_cols) >= 2:
            return "scatter_plot"
        else:
            return "table"
    
    async def _create_visualization_metadata(self, data: pd.DataFrame, viz_type: str) -> Dict[str, Any]:
        """Create metadata for visualization configuration"""
        
        numeric_cols = list(data.select_dtypes(include=['number']).columns)
        categorical_cols = list(data.select_dtypes(include=['object', 'category']).columns)
        
        metadata = {
            'chart_type': viz_type,
            'title': 'Business Intelligence Dashboard',
            'x_axis': categorical_cols[0] if categorical_cols else numeric_cols[0] if numeric_cols else data.columns[0],
            'y_axis': numeric_cols[0] if numeric_cols else data.columns[1] if len(data.columns) > 1 else data.columns[0],
            'color_by': categorical_cols[1] if len(categorical_cols) > 1 else None,
            'guidelines': self._get_viz_guidelines(viz_type)
        }
        
        # Specific configurations by chart type
        if viz_type == "time_series":
            metadata['chart_type'] = "Line Chart"
            metadata['guidelines'] = "Use for showing trends over time. Connect data points with lines."
        elif viz_type == "bar_chart":
            metadata['chart_type'] = "Vertical Bar Chart"
            metadata['guidelines'] = "Use for comparing categories. Sort by value for clarity."
        elif viz_type == "horizontal_bar":
            metadata['chart_type'] = "Horizontal Bar Chart"
            metadata['guidelines'] = "Use for many categories or long category names."
        elif viz_type == "grouped_bar":
            metadata['chart_type'] = "Grouped Bar Chart"
            metadata['guidelines'] = "Use for comparing multiple measures across categories."
        elif viz_type == "scatter_plot":
            metadata['chart_type'] = "Scatter Plot"
            metadata['guidelines'] = "Use for showing relationships between numeric variables."
        else:
            metadata['chart_type'] = "Data Table"
            metadata['guidelines'] = "Use for detailed data examination and exact values."
        
        return metadata
    
    def _get_viz_guidelines(self, viz_type: str) -> str:
        """Get visualization best practices"""
        guidelines = {
            'time_series': "Show clear time progression, use consistent intervals, highlight key trends",
            'bar_chart': "Sort by value, use consistent colors, include data labels for clarity",
            'horizontal_bar': "Good for long category names, sort by value, ensure readability",
            'grouped_bar': "Use distinct colors, include legend, avoid too many groups",
            'scatter_plot': "Show correlation, consider adding trend line, use appropriate scales",
            'table': "Include sorting options, highlight key metrics, use conditional formatting"
        }
        return guidelines.get(viz_type, "Follow data visualization best practices")
    
    async def _publish_to_tableau(self, data: pd.DataFrame, viz_metadata: Dict[str, Any]) -> Optional[str]:
        """Publish data and visualization to Tableau"""
        try:
            # Mock Tableau publishing - replace with actual Tableau Server REST API calls
            datasource_name = f"bi_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create extract or live connection
            extract_info = {
                'name': datasource_name,
                'data_rows': len(data),
                'columns': list(data.columns),
                'created_at': datetime.now().isoformat()
            }
            
            # Mock URL - replace with actual Tableau publishing
            tableau_url = f"{self.tableau_config['server_url']}/datasources/{datasource_name}"
            
            self.logger.info(f"Published to Tableau: {tableau_url}")
            return tableau_url
            
        except Exception as e:
            self.logger.error(f"Tableau publishing error: {e}")
            return None
    
    async def _publish_to_powerbi(self, data: pd.DataFrame, viz_metadata: Dict[str, Any]) -> Optional[str]:
        """Publish data and visualization to Power BI"""
        try:
            # Mock Power BI publishing - replace with actual Power BI REST API calls
            dataset_name = f"bi_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create dataset
            dataset_info = {
                'name': dataset_name,
                'tables': [{
                    'name': 'QueryResults',
                    'columns': [{'name': col, 'dataType': self._map_powerbi_type(data[col].dtype)} 
                               for col in data.columns]
                }]
            }
            
            # Mock URL - replace with actual Power BI publishing
            powerbi_url = f"https://app.powerbi.com/datasets/{dataset_name}"
            
            self.logger.info(f"Published to Power BI: {powerbi_url}")
            return powerbi_url
            
        except Exception as e:
            self.logger.error(f"Power BI publishing error: {e}")
            return None
    
    def _map_powerbi_type(self, pandas_dtype) -> str:
        """Map pandas dtypes to Power BI data types"""
        type_mapping = {
            'int64': 'Int64',
            'float64': 'Double',
            'object': 'String',
            'bool': 'Boolean',
            'datetime64[ns]': 'DateTime'
        }
        return type_mapping.get(str(pandas_dtype), 'String')


# =============================================================================
# ALERTING & KPI AGENT
# =============================================================================

class AlertingAgent(BaseAgent):
    """Handles KPI monitoring and alerting"""
    
    def __init__(self, workspace_client: WorkspaceClient, security_manager: SecurityManager):
        super().__init__(AgentType.ALERTING, workspace_client)
        self.security_manager = security_manager
        self.kpi_definitions = {}
        self.alert_history = []
    
    async def process(self, request: QueryRequest) -> QueryResult:
        """Process alerting requests"""
        start_time = time.time()
        
        try:
            if request.request_type == RequestType.ALERT_SETUP:
                result = await self._setup_alert(request)
            elif request.request_type == RequestType.ALERT_CHECK:
                result = await self._check_alerts(request)
            else:
                result = self._create_result(
                    request, False,
                    error_message=f"Unsupported alerting request type: {request.request_type}"
                )
            
            result.execution_time_ms = int((time.time() - start_time) * 1000)
            return result
            
        except Exception as e:
            self.logger.error(f"Alerting error: {e}")
            return self._create_result(
                request, False,
                error_message=f"Alerting failed: {str(e)}",
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def _setup_alert(self, request: QueryRequest) -> QueryResult:
        """Set up a new KPI alert"""
        try:
            # Parse alert setup request
            alert_config = await self._parse_alert_request(request.natural_language_query)
            
            # Generate SQL for KPI monitoring
            kpi_sql = await self._generate_kpi_sql(alert_config)
            
            # Validate SQL safety
            is_safe, safety_error = self.security_manager.validate_sql_safety(kpi_sql)
            if not is_safe:
                return self._create_result(
                    request, False,
                    error_message=f"KPI SQL validation failed: {safety_error}"
                )
            
            # Create KPI definition
            kpi_def = KPIDefinition(
                kpi_id=str(uuid4()),
                name=alert_config['name'],
                sql_query=kpi_sql,
                threshold_value=alert_config['threshold'],
                comparison_operator=alert_config['operator'],
                alert_recipients=alert_config['recipients'],
                check_frequency=alert_config['frequency'],
                created_by=request.user_context.user_id
            )
            
            # Store KPI definition
            self.kpi_definitions[kpi_def.kpi_id] = kpi_def
            
            explanation = f"""
**Alert Setup Complete:**
- KPI Name: {kpi_def.name}
- Threshold: {kpi_def.comparison_operator} {kpi_def.threshold_value}
- Check Frequency: {kpi_def.check_frequency}
- Recipients: {', '.join(kpi_def.alert_recipients)}

**Monitoring Query:**
```sql
{kpi_sql}
```

Alert ID: {kpi_def.kpi_id}
"""
            
            return self._create_result(
                request, True,
                explanation=explanation,
                sql_query=kpi_sql,
                metadata={
                    'kpi_id': kpi_def.kpi_id,
                    'alert_config': alert_config
                }
            )
            
        except Exception as e:
            return self._create_result(
                request, False,
                error_message=f"Alert setup failed: {str(e)}"
            )
    
    async def _check_alerts(self, request: QueryRequest) -> QueryResult:
        """Check all active alerts and send notifications"""
        try:
            alerts_triggered = []
            
            for kpi_id, kpi_def in self.kpi_definitions.items():
                if not kpi_def.enabled:
                    continue
                
                # Execute KPI query
                try:
                    # Mock execution - replace with actual SQL execution
                    current_value = await self._execute_kpi_query(kpi_def.sql_query)
                    
                    # Check threshold
                    threshold_breached = self._evaluate_threshold(
                        current_value, kpi_def.threshold_value, kpi_def.comparison_operator
                    )
                    
                    if threshold_breached:
                        alert_info = {
                            'kpi_id': kpi_id,
                            'kpi_name': kpi_def.name,
                            'current_value': current_value,
                            'threshold_value': kpi_def.threshold_value,
                            'operator': kpi_def.comparison_operator,
                            'timestamp': datetime.now()
                        }
                        
                        alerts_triggered.append(alert_info)
                        
                        # Send alert
                        await self._send_alert(kpi_def, alert_info)
                    
                    # Update KPI state
                    kpi_def.last_value = current_value
                    kpi_def.last_checked = datetime.now()
                    
                except Exception as e:
                    self.logger.error(f"KPI check error for {kpi_id}: {e}")
            
            explanation = f"""
**Alert Check Complete:**
- Total KPIs Monitored: {len([k for k in self.kpi_definitions.values() if k.enabled])}
- Alerts Triggered: {len(alerts_triggered)}

**Triggered Alerts:**
"""
            
            for alert in alerts_triggered:
                explanation += f"""
- {alert['kpi_name']}: {alert['current_value']} {alert['operator']} {alert['threshold_value']}
"""
            
            if not alerts_triggered:
                explanation += "No thresholds breached. All KPIs within normal ranges."
            
            return self._create_result(
                request, True,
                explanation=explanation,
                metadata={
                    'alerts_triggered': alerts_triggered,
                    'kpis_checked': len(self.kpi_definitions)
                }
            )
            
        except Exception as e:
            return self._create_result(
                request, False,
                error_message=f"Alert checking failed: {str(e)}"
            )
    
    async def _parse_alert_request(self, nl_request: str) -> Dict[str, Any]:
        """Parse natural language alert setup request"""
        
        # Extract threshold information
        threshold_patterns = [
            r'(?:if|when|alert)\s+(\w+)\s*([><=!]+)\s*(\d+(?:\.\d+)?)',
            r'(\w+)\s+(?:above|below|exceeds|drops below)\s+(\d+(?:\.\d+)?)',
            r'(?:churn|rate)\s*>\s*(\d+(?:\.\d+)?)'
        ]
        
        threshold_value = 3.0  # Default
        operator = ">"  # Default
        metric_name = "churn_rate"  # Default
        
        for pattern in threshold_patterns:
            match = re.search(pattern, nl_request.lower())
            if match:
                if len(match.groups()) == 3:
                    metric_name, operator, threshold_value = match.groups()
                    threshold_value = float(threshold_value)
                elif len(match.groups()) == 2:
                    threshold_value = float(match.groups()[1])
                    if 'above' in nl_request.lower() or 'exceeds' in nl_request.lower():
                        operator = ">"
                    elif 'below' in nl_request.lower():
                        operator = "<"
                break
        
        # Extract frequency
        frequency = "0 */6 * * *"  # Default: every 6 hours
        if 'daily' in nl_request.lower():
            frequency = "0 9 * * *"  # 9 AM daily
        elif 'hourly' in nl_request.lower():
            frequency = "0 * * * *"  # Every hour
        elif 'weekly' in nl_request.lower():
            frequency = "0 9 * * 1"  # Monday 9 AM
        
        return {
            'name': f"{metric_name.title()} Monitor",
            'metric': metric_name,
            'threshold': threshold_value,
            'operator': operator,
            'frequency': frequency,
            'recipients': ['user@company.com']  # Default - extract from context
        }
    
    async def _generate_kpi_sql(self, alert_config: Dict[str, Any]) -> str:
        """Generate SQL for KPI monitoring"""
        
        metric = alert_config['metric']
        
        if 'churn' in metric:
            return """
            SELECT 
                AVG(churn_rate) * 100 as current_churn_rate
            FROM customers.customers 
            WHERE signup_date >= DATE_SUB(CURRENT_DATE(), 7)
            """
        elif 'revenue' in metric:
            return """
            SELECT 
                SUM(amount) as current_revenue
            FROM sales.revenue 
            WHERE created_date >= DATE_SUB(CURRENT_DATE(), 1)
            """
        else:
            return f"SELECT COUNT(*) as {metric}_count FROM default.metrics WHERE date = CURRENT_DATE()"
    
    async def _execute_kpi_query(self, sql_query: str) -> float:
        """Execute KPI monitoring query"""
        # Mock execution - replace with actual Databricks SQL execution
        if 'churn_rate' in sql_query:
            return 3.2  # Mock churn rate
        elif 'revenue' in sql_query:
            return 125000.0  # Mock revenue
        else:
            return 42.0  # Mock generic metric
    
    def _evaluate_threshold(self, current_value: float, threshold: float, operator: str) -> bool:
        """Evaluate if threshold is breached"""
        if operator == ">":
            return current_value > threshold
        elif operator == "<":
            return current_value < threshold
        elif operator == ">=":
            return current_value >= threshold
        elif operator == "<=":
            return current_value <= threshold
        elif operator == "==":
            return current_value == threshold
        elif operator == "!=":
            return current_value != threshold
        else:
            return False
    
    async def _send_alert(self, kpi_def: KPIDefinition, alert_info: Dict[str, Any]):
        """Send alert notification"""
        try:
            # Compose alert message
            message = f"""
🚨 KPI Alert: {kpi_def.name}

Current Value: {alert_info['current_value']:.2f}
Threshold: {alert_info['operator']} {alert_info['threshold_value']:.2f}
Timestamp: {alert_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

Query: {kpi_def.sql_query}

This is an automated alert from the BI Analytics Platform.
"""
            
            # Mock sending - replace with actual email/Slack/Teams integration
            for recipient in kpi_def.alert_recipients:
                self.logger.info(f"Sending alert to {recipient}: {kpi_def.name}")
                # await self._send_email(recipient, f"KPI Alert: {kpi_def.name}", message)
                # await self._send_slack(recipient, message)
            
            # Log alert
            self.alert_history.append({
                'kpi_id': kpi_def.kpi_id,
                'alert_info': alert_info,
                'sent_at': datetime.now(),
                'recipients': kpi_def.alert_recipients
            })
            
        except Exception as e:
            self.logger.error(f"Alert sending error: {e}")


# =============================================================================
# MAIN APPLICATION CLASS
# =============================================================================

class MultiAgentBIPlatform:
    """Main application orchestrating all agents"""
    
    def __init__(self, databricks_config: Dict[str, Any]):
        # Initialize Databricks workspace client
        self.workspace_client = WorkspaceClient(
            host=databricks_config.get('host'),
            token=databricks_config.get('token')
        )
        
        # Initialize security manager
        self.security_manager = SecurityManager(self.workspace_client)
        
        # Initialize orchestrator
        self.orchestrator = OrchestratorAgent(self.workspace_client, self.security_manager)
        
        # Initialize and register agents
        self._initialize_agents()
        
        # Setup logging
        self._setup_logging()
        
        self.logger = logging.getLogger('bi_platform')
        self.logger.info("Multi-Agent BI Platform initialized")
    
    def _initialize_agents(self):
        """Initialize and register all agents"""
        
        # Create agents
        nl_sql_agent = NLToSQLAgent(self.workspace_client, self.security_manager)
        analysis_agent = AnalysisAgent(self.workspace_client)
        visualization_agent = VisualizationAgent(self.workspace_client)
        alerting_agent = AlertingAgent(self.workspace_client, self.security_manager)
        
        # Register with orchestrator
        self.orchestrator.register_agent(AgentType.NL_SQL, nl_sql_agent)
        self.orchestrator.register_agent(AgentType.ANALYSIS, analysis_agent)
        self.orchestrator.register_agent(AgentType.VISUALIZATION, visualization_agent)
        self.orchestrator.register_agent(AgentType.ALERTING, alerting_agent)
    
    def _setup_logging(self):
        """Configure logging for the platform"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('bi_platform.log')
            ]
        )
    
    async def ask_question(self, user_context: UserContext, question: str, 
                          include_analysis: bool = False, create_visualization: bool = False) -> QueryResult:
        """Main entry point for natural language questions"""
        
        request = QueryRequest(
            request_id=str(uuid4()),
            user_context=user_context,
            request_type=RequestType.QUERY,
            natural_language_query=question,
            metadata={
                'include_analysis': include_analysis,
                'create_visualization': create_visualization
            }
        )
        
        self.logger.info(f"Processing question: {question} (User: {user_context.user_id})")
        
        # Process through orchestrator
        result = await self.orchestrator.process(request)
        
        # If visualization requested, create it
        if create_visualization and result.success and result.data is not None:
            viz_request = QueryRequest(
                request_id=f"{request.request_id}_viz",
                user_context=user_context,
                request_type=RequestType.VISUALIZATION,
                natural_language_query=f"Visualize: {question}",
                metadata={'source_data': result.data}
            )
            
            viz_result = await self.orchestrator.process(viz_request)
            if viz_result.success:
                result.visualization_url = viz_result.visualization_url
                result.explanation += f"\n\n{viz_result.explanation}"
        
        return result
    
    async def setup_alert(self, user_context: UserContext, alert_description: str) -> QueryResult:
        """Set up a new KPI alert"""
        
        request = QueryRequest(
            request_id=str(uuid4()),
            user_context=user_context,
            request_type=RequestType.ALERT_SETUP,
            natural_language_query=alert_description
        )
        
        self.logger.info(f"Setting up alert: {alert_description} (User: {user_context.user_id})")
        
        return await self.orchestrator.process(request)
    
    async def check_alerts(self, user_context: UserContext) -> QueryResult:
        """Check all active alerts"""
        
        request = QueryRequest(
            request_id=str(uuid4()),
            user_context=user_context,
            request_type=RequestType.ALERT_CHECK,
            natural_language_query="Check all alerts"
        )
        
        return await self.orchestrator.process(request)
    
    def get_audit_log(self, user_context: UserContext) -> List[Dict[str, Any]]:
        """Get audit log (admin only)"""
        if 'admin' not in user_context.roles:
            raise PermissionError("Insufficient privileges to access audit log")
        
        return self.security_manager.audit_log


# =============================================================================
# EXAMPLE USAGE & DEMONSTRATION
# =============================================================================

async def demo_platform():
    """Demonstrate the multi-agent BI platform"""
    
    # Configuration
    databricks_config = {
        'host': 'https://your-databricks-workspace.cloud.databricks.com',
        'token': 'your-databricks-token'
    }
    
    # Initialize platform
    platform = MultiAgentBIplatform(databricks_config)
    
    # Create user context
    user_context = UserContext(
        user_id="demo_user",
        email="user@company.com",
        department="Sales",
        roles=["analyst", "viewer"],
        security_clearance=SecurityLevel.INTERNAL,
        accessible_schemas=["sales", "customers", "hr"],
        accessible_tables=["sales.revenue", "customers.customers", "hr.employees"]
    )
    
    print("🚀 Multi-Agent BI Platform Demo")
    print("=" * 50)
    
    # Demo 1: Natural Language Query
    print("\n📊 Demo 1: Natural Language Query")
    print("-" * 30)
    
    question1 = "What was Q2 2024 revenue by region?"
    print(f"Question: {question1}")
    
    result1 = await platform.ask_question(
        user_context, question1, 
        include_analysis=True, 
        create_visualization=True
    )
    
    if result1.success:
        print("✅ Query successful!")
        print(f"📈 Data: {len(result1.data)} rows returned")
        print(f"⏱️  Execution time: {result1.execution_time_ms}ms")
        print(f"📝 Explanation:\n{result1.explanation}")
        if result1.visualization_url:
            print(f"📊 Visualization: {result1.visualization_url}")
    else:
        print(f"❌ Query failed: {result1.error_message}")
    
    # Demo 2: Analysis Request
    print("\n🔍 Demo 2: Advanced Analysis")
    print("-" * 30)
    
    question2 = "Find month-over-month growth and any outliers for subscriptions"
    print(f"Question: {question2}")
    
    result2 = await platform.ask_question(user_context, question2, include_analysis=True)
    
    if result2.success:
        print("✅ Analysis successful!")
        print(f"📊 Analysis:\n{result2.explanation}")
    else:
        print(f"❌ Analysis failed: {result2.error_message}")
    
    # Demo 3: Alert Setup
    print("\n🚨 Demo 3: Alert Setup")
    print("-" * 30)
    
    alert_description = "Alert me if churn rate exceeds 3% week-over-week"
    print(f"Alert: {alert_description}")
    
    alert_result = await platform.setup_alert(user_context, alert_description)
    
    if alert_result.success:
        print("✅ Alert setup successful!")
        print(f"📋 Configuration:\n{alert_result.explanation}")
    else:
        print(f"❌ Alert setup failed: {alert_result.error_message}")
    
    # Demo 4: Alert Check
    print("\n⏰ Demo 4: Alert Monitoring")
    print("-" * 30)
    
    check_result = await platform.check_alerts(user_context)
    
    if check_result.success:
        print("✅ Alert check completed!")
        print(f"📊 Results:\n{check_result.explanation}")
    else:
        print(f"❌ Alert check failed: {check_result.error_message}")
    
    # Demo 5: Security & Audit
    print("\n🔒 Demo 5: Security & Governance")
    print("-" * 30)
    
    try:
        audit_log = platform.get_audit_log(user_context)
        print(f"📋 Audit entries: {len(audit_log)}")
        for entry in audit_log[-3:]:  # Show last 3 entries
            print(f"  • {entry['timestamp']}: {entry['event_type']} - {entry['details']}")
    except PermissionError:
        print("❌ Insufficient privileges for audit log access")
    
    print("\n🎉 Demo completed!")
    return platform


# =============================================================================
# CONFIGURATION & DEPLOYMENT
# =============================================================================

class PlatformConfig:
    """Configuration management for the BI platform"""
    
    @staticmethod
    def load_databricks_config() -> Dict[str, Any]:
        """Load Databricks configuration"""
        return {
            'host': 'https://your-workspace.cloud.databricks.com',
            'token': 'your-access-token',
            'warehouse_id': 'your-sql-warehouse-id',
            'catalog': 'your-unity-catalog',
            'schema': 'your-default-schema'
        }
    
    @staticmethod
    def load_security_config() -> Dict[str, Any]:
        """Load security configuration"""
        return {
            'unity_catalog_enabled': True,
            'pii_masking_enabled': True,
            'audit_logging_enabled': True,
            'row_level_security': True,
            'column_level_security': True,
            'query_timeout_seconds': 300,
            'max_result_rows': 10000
        }
    
    @staticmethod
    def load_integration_config() -> Dict[str, Any]:
        """Load external integration configuration"""
        return {
            'tableau': {
                'server_url': 'https://your-tableau-server.com',
                'site_id': 'your-site',
                'username': 'service_account',
                'project_name': 'BI_Analytics'
            },
            'powerbi': {
                'tenant_id': 'your-tenant-id',
                'client_id': 'your-app-id',
                'workspace_id': 'your-workspace-id'
            },
            'notifications': {
                'email': {
                    'smtp_server': 'smtp.company.com',
                    'smtp_port': 587,
                    'username': 'noreply@company.com'
                },
                'slack': {
                    'webhook_url': 'https://hooks.slack.com/your-webhook',
                    'channel': '#bi-alerts'
                },
                'teams': {
                    'webhook_url': 'https://your-org.webhook.office.com/your-webhook'
                }
            }
        }


# =============================================================================
# PRODUCTION DEPLOYMENT UTILITIES
# =============================================================================

class ProductionDeployment:
    """Utilities for production deployment"""
    
    @staticmethod
    def create_databricks_job() -> Dict[str, Any]:
        """Create Databricks job configuration for the platform"""
        return {
            "name": "Multi-Agent-BI-Platform",
            "email_notifications": {
                "on_failure": ["admin@company.com"],
                "on_success": [],
                "no_alert_for_skipped_runs": False
            },
            "webhook_notifications": {},
            "timeout_seconds": 3600,
            "max_concurrent_runs": 1,
            "tasks": [
                {
                    "task_key": "bi_platform_server",
                    "description": "Main BI platform server",
                    "python_wheel_task": {
                        "package_name": "bi_platform",
                        "entry_point": "main"
                    },
                    "job_cluster_key": "bi_cluster",
                    "timeout_seconds": 0,
                    "email_notifications": {}
                },
                {
                    "task_key": "alert_scheduler",
                    "description": "KPI alert monitoring",
                    "depends_on": [{"task_key": "bi_platform_server"}],
                    "python_wheel_task": {
                        "package_name": "bi_platform",
                        "entry_point": "alert_scheduler"
                    },
                    "job_cluster_key": "bi_cluster"
                }
            ],
            "job_clusters": [
                {
                    "job_cluster_key": "bi_cluster",
                    "new_cluster": {
                        "spark_version": "13.3.x-scala2.12",
                        "node_type_id": "i3.xlarge",
                        "driver_node_type_id": "i3.xlarge",
                        "num_workers": 2,
                        "cluster_log_conf": {
                            "dbfs": {"destination": "dbfs:/cluster-logs"}
                        },
                        "init_scripts": [],
                        "enable_elastic_disk": False,
                        "disk_spec": {},
                        "cluster_source": "JOB"
                    }
                }
            ],
            "format": "MULTI_TASK"
        }
    
    @staticmethod
    def create_unity_catalog_setup() -> List[str]:
        """Generate Unity Catalog setup SQL"""
        return [
            # Create catalogs
            "CREATE CATALOG IF NOT EXISTS bi_analytics_prod",
            "CREATE CATALOG IF NOT EXISTS bi_analytics_dev",
            
            # Create schemas
            "CREATE SCHEMA IF NOT EXISTS bi_analytics_prod.sales",
            "CREATE SCHEMA IF NOT EXISTS bi_analytics_prod.customers", 
            "CREATE SCHEMA IF NOT EXISTS bi_analytics_prod.hr",
            "CREATE SCHEMA IF NOT EXISTS bi_analytics_prod.finance",
            
            # Create service principal
            "CREATE SERVICE PRINCIPAL IF NOT EXISTS 'bi-platform-service'",
            
            # Grant permissions
            "GRANT USE CATALOG ON CATALOG bi_analytics_prod TO 'bi-platform-service'",
            "GRANT USE SCHEMA ON SCHEMA bi_analytics_prod.sales TO 'bi-platform-service'",
            "GRANT SELECT ON SCHEMA bi_analytics_prod.sales TO 'bi-platform-service'",
            
            # Create security policies
            """CREATE OR REPLACE FUNCTION bi_analytics_prod.security.mask_pii(input_value STRING)
               RETURNS STRING
               LANGUAGE SQL
               AS $
                 CASE 
                   WHEN input_value RLIKE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}'
                   THEN '[MASKED_EMAIL]'
                   ELSE input_value
                 END
               $""",
            
            # Row-level security example
            """CREATE OR REPLACE FUNCTION bi_analytics_prod.security.user_filter()
               RETURNS STRING
               AS $
                 CASE 
                   WHEN IS_MEMBER('admin') THEN '1=1'
                   WHEN IS_MEMBER('sales_team') THEN 'department = "Sales"'
                   ELSE '1=0'
                 END
               $"""
        ]
    
    @staticmethod
    def create_monitoring_dashboard() -> Dict[str, Any]:
        """Create monitoring dashboard configuration"""
        return {
            "dashboard_name": "BI Platform Monitoring",
            "refresh_schedule": "*/5 * * * *",  # Every 5 minutes
            "widgets": [
                {
                    "name": "Query Volume",
                    "type": "line_chart",
                    "query": """
                        SELECT 
                            DATE_TRUNC('hour', timestamp) as hour,
                            COUNT(*) as query_count
                        FROM bi_platform_logs.query_metrics 
                        WHERE timestamp >= CURRENT_TIMESTAMP() - INTERVAL 24 HOURS
                        GROUP BY DATE_TRUNC('hour', timestamp)
                        ORDER BY hour
                    """
                },
                {
                    "name": "Success Rate",
                    "type": "gauge",
                    "query": """
                        SELECT 
                            (SUM(CASE WHEN success = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as success_rate
                        FROM bi_platform_logs.query_metrics 
                        WHERE timestamp >= CURRENT_TIMESTAMP() - INTERVAL 1 HOUR
                    """
                },
                {
                    "name": "Response Time",
                    "type": "histogram",
                    "query": """
                        SELECT 
                            execution_time_ms,
                            COUNT(*) as frequency
                        FROM bi_platform_logs.query_metrics 
                        WHERE timestamp >= CURRENT_TIMESTAMP() - INTERVAL 24 HOURS
                        GROUP BY execution_time_ms
                        ORDER BY execution_time_ms
                    """
                },
                {
                    "name": "Active Alerts",
                    "type": "table",
                    "query": """
                        SELECT 
                            kpi_name,
                            last_value,
                            threshold_value,
                            last_checked
                        FROM bi_platform_logs.kpi_status 
                        WHERE enabled = true
                        ORDER BY last_checked DESC
                    """
                }
            ]
        }


# =============================================================================
# REST API INTERFACE
# =============================================================================

class BIPlatformAPI:
    """REST API interface for the BI platform"""
    
    def __init__(self, platform: MultiAgentBIPlatform):
        self.platform = platform
    
    async def query_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """REST endpoint for natural language queries"""
        try:
            # Extract user context
            user_context = UserContext(**request_data['user_context'])
            
            # Process query
            result = await self.platform.ask_question(
                user_context,
                request_data['question'],
                request_data.get('include_analysis', False),
                request_data.get('create_visualization', False)
            )
            
            # Convert result to JSON-serializable format
            return {
                'request_id': result.request_id,
                'success': result.success,
                'data': result.data.to_dict('records') if result.data is not None else None,
                'sql_query': result.sql_query,
                'explanation': result.explanation,
                'visualization_url': result.visualization_url,
                'error_message': result.error_message,
                'execution_time_ms': result.execution_time_ms,
                'metadata': result.metadata
            }
            
        except Exception as e:
            return {
                'success': False,
                'error_message': f"API error: {str(e)}"
            }
    
    async def alert_setup_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """REST endpoint for alert setup"""
        try:
            user_context = UserContext(**request_data['user_context'])
            
            result = await self.platform.setup_alert(
                user_context,
                request_data['alert_description']
            )
            
            return {
                'request_id': result.request_id,
                'success': result.success,
                'explanation': result.explanation,
                'error_message': result.error_message,
                'metadata': result.metadata
            }
            
        except Exception as e:
            return {
                'success': False,
                'error_message': f"Alert setup error: {str(e)}"
            }
    
    async def health_check_endpoint(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'agents': {
                agent_type.value: 'active' 
                for agent_type in self.platform.orchestrator.agents.keys()
            },
            'version': '1.0.0'
        }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Example configuration loading
    config = {
        'databricks': PlatformConfig.load_databricks_config(),
        'security': PlatformConfig.load_security_config(),
        'integrations': PlatformConfig.load_integration_config()
    }
    
    # Initialize platform
    async def main():
        try:
            # Run demonstration
            platform = await demo_platform()
            
            # Example of running the platform as a service
            print("\n🔄 Platform ready for production use")
            print("Available endpoints:")
            print("  • POST /api/v1/query - Natural language queries")
            print("  • POST /api/v1/alerts/setup - Alert configuration")
            print("  • GET /api/v1/alerts/check - Alert monitoring")
            print("  • GET /api/v1/health - Health check")
            print("  • GET /api/v1/audit - Audit log (admin only)")
            
            # Initialize API
            api = BIPlatformAPI(platform)
            
            # Example API usage
            print("\n📡 Example API Request:")
            sample_request = {
                'user_context': {
                    'user_id': 'api_user',
                    'email': 'api@company.com',
                    'department': 'Finance',
                    'roles': ['analyst'],
                    'security_clearance': 'internal',
                    'accessible_schemas': ['sales', 'finance'],
                    'accessible_tables': ['sales.revenue', 'finance.expenses']
                },
                'question': 'Show me revenue trends for the last quarter',
                'include_analysis': True,
                'create_visualization': True
            }
            
            api_result = await api.query_endpoint(sample_request)
            print(f"API Response: {json.dumps(api_result, indent=2, default=str)}")
            
        except Exception as e:
            print(f"❌ Platform initialization error: {e}")
    
    # Run the demonstration
    asyncio.run(main())


# =============================================================================
# PRODUCTION CHECKLIST
# =============================================================================

"""
PRODUCTION DEPLOYMENT CHECKLIST:

🔧 INFRASTRUCTURE:
□ Configure Databricks workspace with appropriate compute resources
□ Set up Unity Catalog with proper governance policies
□ Configure SQL warehouses for query execution
□ Set up monitoring and alerting for platform health

🔐 SECURITY:
□ Implement proper authentication (OAuth 2.0, SAML)
□ Configure row-level and column-level security policies
□ Set up PII detection and masking rules
□ Enable comprehensive audit logging
□ Implement rate limiting and query timeouts

📊 INTEGRATIONS:
□ Configure Tableau Server/Online REST API connections
□ Set up Power BI service principal and workspace access
□ Configure SMTP/email service for alerts
□ Set up Slack/Teams webhook integrations
□ Test all external service connections

🚀 DEPLOYMENT:
□ Package application as Databricks wheel/egg
□ Configure job clusters with appropriate libraries
□ Set up CI/CD pipeline for updates
□ Configure environment-specific settings (dev/staging/prod)
□ Set up backup and disaster recovery procedures

📈 MONITORING:
□ Set up application performance monitoring
□ Configure query performance tracking
□ Set up cost monitoring and optimization alerts
□ Implement user usage analytics
□ Configure platform health dashboards

🧪 TESTING:
□ Unit tests for all agent components
□ Integration tests for external services
□ Load testing for concurrent users
□ Security penetration testing
□ End-to-end user acceptance testing

📋 DOCUMENTATION:
□ API documentation and examples
□ User guides for natural language querying
□ Admin guides for platform management
□ Security policies and procedures
□ Incident response procedures

🎯 PERFORMANCE TARGETS:
□ Sub-10 second response time for 90% of queries
□ 99.9% uptime SLA
□ Support for 100+ concurrent users
□ Query result caching for improved performance
□ Automatic scaling based on demand
"""
