"""
System Performance Monitor

Enhanced performance monitoring system that provides comprehensive response quality measurement,
confidence score tracking and analysis, and system health monitoring with alerting capabilities.
This extends the existing analytics manager with advanced performance monitoring features.
"""

import time
import statistics
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import threading
import json
import os

from ..interfaces.base import (
    AnalyticsManagerInterface, ProcessedQuery, Response, FAQEntry
)
from ..utils.logging import get_rag_logger


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PerformanceAlert:
    """Data model for performance alerts."""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    component: str
    metric: str
    current_value: float
    threshold: float
    message: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class ResponseQualityMetrics:
    """Data model for response quality metrics."""
    confidence_score: float
    relevance_score: float
    completeness_score: float
    coherence_score: float
    source_attribution_score: float
    response_time: float
    user_satisfaction: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemHealthMetrics:
    """Data model for system health metrics."""
    component_name: str
    status: str  # healthy, degraded, unhealthy
    response_time: float
    error_rate: float
    throughput: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceMonitor:
    """
    Advanced performance monitoring system that provides comprehensive metrics,
    alerting, and health monitoring capabilities for the RAG system.
    
    Features:
    - Response quality measurement with multiple dimensions
    - Confidence score tracking and trend analysis
    - System health monitoring with real-time alerting
    - Performance threshold management
    - Automated alert generation and resolution
    """

    def __init__(self, 
                 analytics_manager: Optional[AnalyticsManagerInterface] = None,
                 storage_path: Optional[str] = None,
                 alert_callbacks: Optional[List[Callable]] = None):
        """
        Initialize the performance monitor.
        
        Args:
            analytics_manager: Optional analytics manager for integration
            storage_path: Path for storing performance data
            alert_callbacks: List of callback functions for alerts
        """
        self.logger = get_rag_logger('performance_monitor')
        self.analytics_manager = analytics_manager
        self.storage_path = storage_path or "performance_data"
        self.alert_callbacks = alert_callbacks or []
        
        # Performance metrics storage
        self.response_quality_metrics: deque = deque(maxlen=10000)
        self.system_health_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.confidence_score_history: deque = deque(maxlen=5000)
        
        # Real-time monitoring
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []
        self.monitoring_active = True
        self.monitoring_thread = None
        
        # Performance thresholds
        self.thresholds = {
            'response_time': {'warning': 2.0, 'critical': 5.0},
            'confidence_score': {'warning': 0.5, 'critical': 0.3},
            'error_rate': {'warning': 0.05, 'critical': 0.15},
            'system_health': {'warning': 0.8, 'critical': 0.6},
            'memory_usage': {'warning': 0.8, 'critical': 0.95},
            'throughput': {'warning': 10.0, 'critical': 5.0}  # queries per minute
        }
        
        # Trend analysis
        self.trend_windows = {
            'short_term': timedelta(minutes=15),
            'medium_term': timedelta(hours=1),
            'long_term': timedelta(hours=24)
        }
        
        # Initialize storage and monitoring
        self._ensure_storage_directory()
        self._load_persistent_data()
        self._start_monitoring_thread()
        
        self.logger.info("Performance Monitor initialized with comprehensive tracking")

    def _ensure_storage_directory(self) -> None:
        """Ensure performance data storage directory exists."""
        try:
            os.makedirs(self.storage_path, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Failed to create performance storage directory: {e}")

    def _load_persistent_data(self) -> None:
        """Load persistent performance data from storage."""
        try:
            # Load alert history
            alerts_file = os.path.join(self.storage_path, "alert_history.json")
            if os.path.exists(alerts_file):
                with open(alerts_file, 'r') as f:
                    alert_data = json.load(f)
                    for alert_dict in alert_data:
                        alert = PerformanceAlert(
                            id=alert_dict['id'],
                            timestamp=datetime.fromisoformat(alert_dict['timestamp']),
                            severity=AlertSeverity(alert_dict['severity']),
                            component=alert_dict['component'],
                            metric=alert_dict['metric'],
                            current_value=alert_dict['current_value'],
                            threshold=alert_dict['threshold'],
                            message=alert_dict['message'],
                            resolved=alert_dict.get('resolved', False),
                            resolved_at=datetime.fromisoformat(alert_dict['resolved_at']) if alert_dict.get('resolved_at') else None
                        )
                        self.alert_history.append(alert)
            
            # Load performance thresholds
            thresholds_file = os.path.join(self.storage_path, "thresholds.json")
            if os.path.exists(thresholds_file):
                with open(thresholds_file, 'r') as f:
                    self.thresholds.update(json.load(f))
            
            self.logger.info("Loaded persistent performance data")
        except Exception as e:
            self.logger.warning(f"Failed to load persistent performance data: {e}")

    def _save_persistent_data(self) -> None:
        """Save performance data to persistent storage."""
        try:
            # Save alert history (keep only recent alerts)
            alerts_file = os.path.join(self.storage_path, "alert_history.json")
            recent_alerts = [
                alert for alert in self.alert_history 
                if alert.timestamp > datetime.now() - timedelta(days=30)
            ]
            alert_data = []
            for alert in recent_alerts:
                alert_dict = {
                    'id': alert.id,
                    'timestamp': alert.timestamp.isoformat(),
                    'severity': alert.severity.value,
                    'component': alert.component,
                    'metric': alert.metric,
                    'current_value': alert.current_value,
                    'threshold': alert.threshold,
                    'message': alert.message,
                    'resolved': alert.resolved,
                    'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None
                }
                alert_data.append(alert_dict)
            
            with open(alerts_file, 'w') as f:
                json.dump(alert_data, f, indent=2)
            
            # Save thresholds
            thresholds_file = os.path.join(self.storage_path, "thresholds.json")
            with open(thresholds_file, 'w') as f:
                json.dump(self.thresholds, f, indent=2)
            
            self.logger.debug("Saved persistent performance data")
        except Exception as e:
            self.logger.warning(f"Failed to save persistent performance data: {e}")

    def _start_monitoring_thread(self) -> None:
        """Start background monitoring thread."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("Started performance monitoring thread")

    def _monitoring_loop(self) -> None:
        """Background monitoring loop for real-time health checks."""
        while self.monitoring_active:
            try:
                # Perform periodic health checks
                self._check_system_health()
                
                # Check for alert conditions
                self._check_alert_conditions()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Save data periodically
                self._save_persistent_data()
                
                # Sleep for monitoring interval
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error

    def measure_response_quality(self, 
                                query: str, 
                                processed_query: ProcessedQuery, 
                                response: Response,
                                response_time: float) -> ResponseQualityMetrics:
        """
        Measure comprehensive response quality metrics.
        
        Args:
            query: Original user query
            processed_query: Processed query object
            response: Generated response
            response_time: Time taken to generate response
            
        Returns:
            ResponseQualityMetrics object with detailed quality scores
        """
        try:
            # Calculate relevance score based on source FAQ matching
            relevance_score = self._calculate_relevance_score(processed_query, response)
            
            # Calculate completeness score based on response content
            completeness_score = self._calculate_completeness_score(query, response)
            
            # Calculate coherence score based on response structure
            coherence_score = self._calculate_coherence_score(response)
            
            # Calculate source attribution score
            source_attribution_score = self._calculate_source_attribution_score(response)
            
            # Create quality metrics
            quality_metrics = ResponseQualityMetrics(
                confidence_score=response.confidence,
                relevance_score=relevance_score,
                completeness_score=completeness_score,
                coherence_score=coherence_score,
                source_attribution_score=source_attribution_score,
                response_time=response_time
            )
            
            # Store metrics
            self.response_quality_metrics.append(quality_metrics)
            self.confidence_score_history.append(response.confidence)
            
            # Check for quality alerts
            self._check_quality_alerts(quality_metrics)
            
            self.logger.debug(f"Response quality measured: relevance={relevance_score:.2f}, "
                            f"completeness={completeness_score:.2f}, coherence={coherence_score:.2f}")
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to measure response quality: {e}")
            # Return basic metrics on error
            return ResponseQualityMetrics(
                confidence_score=response.confidence,
                relevance_score=0.5,
                completeness_score=0.5,
                coherence_score=0.5,
                source_attribution_score=0.5,
                response_time=response_time
            )

    def _calculate_relevance_score(self, processed_query: ProcessedQuery, response: Response) -> float:
        """Calculate relevance score based on query-response matching."""
        try:
            if not response.source_faqs:
                return 0.3  # Low relevance if no sources
            
            # Base score from confidence
            base_score = response.confidence
            
            # Boost for multiple relevant sources
            source_boost = min(0.2, len(response.source_faqs) * 0.05)
            
            # Boost for context usage
            context_boost = 0.1 if response.context_used else 0.0
            
            # Penalty for fallback methods
            method_penalty = 0.2 if response.generation_method == 'fallback' else 0.0
            
            relevance_score = base_score + source_boost + context_boost - method_penalty
            return max(0.0, min(1.0, relevance_score))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate relevance score: {e}")
            return 0.5

    def _calculate_completeness_score(self, query: str, response: Response) -> float:
        """Calculate completeness score based on response content."""
        try:
            # Basic completeness based on response length
            response_length = len(response.text.strip())
            
            if response_length < 20:
                return 0.2  # Very incomplete
            elif response_length < 50:
                return 0.5  # Somewhat incomplete
            elif response_length < 200:
                return 0.8  # Good completeness
            else:
                return 1.0  # Very complete
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate completeness score: {e}")
            return 0.5

    def _calculate_coherence_score(self, response: Response) -> float:
        """Calculate coherence score based on response structure."""
        try:
            text = response.text.strip()
            
            if not text:
                return 0.0
            
            # Basic coherence indicators
            sentences = text.split('.')
            coherence_score = 0.5  # Base score
            
            # Boost for proper sentence structure
            if len(sentences) > 1:
                coherence_score += 0.2
            
            # Boost for source attribution
            if response.source_faqs and any(faq.id in response.metadata.get('sources', []) for faq in response.source_faqs):
                coherence_score += 0.2
            
            # Penalty for very short responses
            if len(text) < 30:
                coherence_score -= 0.3
            
            return max(0.0, min(1.0, coherence_score))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate coherence score: {e}")
            return 0.5

    def _calculate_source_attribution_score(self, response: Response) -> float:
        """Calculate source attribution score."""
        try:
            if not response.source_faqs:
                return 0.0  # No sources to attribute
            
            # Check if sources are properly attributed in metadata
            sources_in_metadata = len(response.metadata.get('sources', []))
            total_sources = len(response.source_faqs)
            
            if total_sources == 0:
                return 0.0
            
            attribution_ratio = sources_in_metadata / total_sources
            return min(1.0, attribution_ratio)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate source attribution score: {e}")
            return 0.5

    def track_confidence_scores(self, window_minutes: int = 60) -> Dict[str, Any]:
        """
        Analyze confidence score patterns and trends.
        
        Args:
            window_minutes: Time window for analysis in minutes
            
        Returns:
            Dictionary containing confidence score analysis
        """
        try:
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            
            # Get recent quality metrics
            recent_metrics = [
                m for m in self.response_quality_metrics 
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return {'error': 'No recent data available'}
            
            confidence_scores = [m.confidence_score for m in recent_metrics]
            
            # Calculate statistics
            analysis = {
                'window_minutes': window_minutes,
                'total_responses': len(confidence_scores),
                'statistics': {
                    'mean': statistics.mean(confidence_scores),
                    'median': statistics.median(confidence_scores),
                    'std_dev': statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0.0,
                    'min': min(confidence_scores),
                    'max': max(confidence_scores)
                },
                'distribution': {
                    'high_confidence': len([s for s in confidence_scores if s > 0.8]),
                    'medium_confidence': len([s for s in confidence_scores if 0.5 <= s <= 0.8]),
                    'low_confidence': len([s for s in confidence_scores if s < 0.5])
                },
                'trends': self._analyze_confidence_trends(confidence_scores),
                'quality_correlation': self._analyze_quality_correlation(recent_metrics)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to track confidence scores: {e}")
            return {'error': str(e)}

    def _analyze_confidence_trends(self, confidence_scores: List[float]) -> Dict[str, Any]:
        """Analyze confidence score trends."""
        try:
            if len(confidence_scores) < 5:
                return {'trend': 'insufficient_data'}
            
            # Simple trend analysis using moving averages
            window_size = min(5, len(confidence_scores) // 3)
            
            if window_size < 2:
                return {'trend': 'insufficient_data'}
            
            # Calculate moving averages
            early_avg = statistics.mean(confidence_scores[:window_size])
            late_avg = statistics.mean(confidence_scores[-window_size:])
            
            trend_direction = 'stable'
            if late_avg > early_avg + 0.05:
                trend_direction = 'improving'
            elif late_avg < early_avg - 0.05:
                trend_direction = 'declining'
            
            return {
                'trend': trend_direction,
                'early_average': early_avg,
                'late_average': late_avg,
                'change': late_avg - early_avg
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze confidence trends: {e}")
            return {'trend': 'error', 'error': str(e)}

    def _analyze_quality_correlation(self, metrics: List[ResponseQualityMetrics]) -> Dict[str, Any]:
        """Analyze correlation between different quality metrics."""
        try:
            if len(metrics) < 3:
                return {'correlation': 'insufficient_data'}
            
            confidence_scores = [m.confidence_score for m in metrics]
            relevance_scores = [m.relevance_score for m in metrics]
            response_times = [m.response_time for m in metrics]
            
            # Simple correlation analysis
            avg_confidence = statistics.mean(confidence_scores)
            avg_relevance = statistics.mean(relevance_scores)
            avg_response_time = statistics.mean(response_times)
            
            return {
                'confidence_relevance_correlation': 'positive' if avg_confidence > 0.7 and avg_relevance > 0.7 else 'mixed',
                'response_time_impact': 'fast' if avg_response_time < 1.0 else 'slow',
                'overall_quality': 'high' if avg_confidence > 0.8 and avg_relevance > 0.8 else 'medium'
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze quality correlation: {e}")
            return {'correlation': 'error', 'error': str(e)}

    def monitor_system_health(self, component_name: str, 
                            status: str, 
                            response_time: float,
                            error_rate: float = 0.0,
                            throughput: float = 0.0) -> SystemHealthMetrics:
        """
        Monitor and record system health metrics for a component.
        
        Args:
            component_name: Name of the system component
            status: Health status (healthy, degraded, unhealthy)
            response_time: Component response time in seconds
            error_rate: Error rate (0.0 to 1.0)
            throughput: Throughput metric (requests per minute)
            
        Returns:
            SystemHealthMetrics object
        """
        try:
            health_metrics = SystemHealthMetrics(
                component_name=component_name,
                status=status,
                response_time=response_time,
                error_rate=error_rate,
                throughput=throughput
            )
            
            # Store metrics
            self.system_health_metrics[component_name].append(health_metrics)
            
            # Check for health alerts
            self._check_health_alerts(health_metrics)
            
            self.logger.debug(f"System health recorded for {component_name}: {status} "
                            f"(response_time={response_time:.2f}s, error_rate={error_rate:.2f})")
            
            return health_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to monitor system health for {component_name}: {e}")
            return SystemHealthMetrics(
                component_name=component_name,
                status='unknown',
                response_time=response_time,
                error_rate=error_rate,
                throughput=throughput
            )

    def _check_system_health(self) -> None:
        """Perform periodic system health checks."""
        try:
            # This would integrate with actual system components
            # For now, we'll simulate basic health monitoring
            
            current_time = datetime.now()
            
            # Check if we have recent health data
            for component_name, metrics_deque in self.system_health_metrics.items():
                if not metrics_deque:
                    continue
                
                latest_metric = metrics_deque[-1]
                time_since_update = current_time - latest_metric.timestamp
                
                # Alert if no updates for too long
                if time_since_update > timedelta(minutes=5):
                    self._create_alert(
                        component=component_name,
                        metric='health_update',
                        current_value=time_since_update.total_seconds(),
                        threshold=300.0,  # 5 minutes
                        severity=AlertSeverity.MEDIUM,
                        message=f"No health updates for {component_name} in {time_since_update}"
                    )
                    
        except Exception as e:
            self.logger.error(f"Error in system health check: {e}")

    def _check_quality_alerts(self, quality_metrics: ResponseQualityMetrics) -> None:
        """Check for quality-related alerts."""
        try:
            # Check confidence score
            if quality_metrics.confidence_score < self.thresholds['confidence_score']['critical']:
                self._create_alert(
                    component='response_generator',
                    metric='confidence_score',
                    current_value=quality_metrics.confidence_score,
                    threshold=self.thresholds['confidence_score']['critical'],
                    severity=AlertSeverity.HIGH,
                    message=f"Critical low confidence score: {quality_metrics.confidence_score:.2f}"
                )
            elif quality_metrics.confidence_score < self.thresholds['confidence_score']['warning']:
                self._create_alert(
                    component='response_generator',
                    metric='confidence_score',
                    current_value=quality_metrics.confidence_score,
                    threshold=self.thresholds['confidence_score']['warning'],
                    severity=AlertSeverity.MEDIUM,
                    message=f"Low confidence score: {quality_metrics.confidence_score:.2f}"
                )
            
            # Check response time
            if quality_metrics.response_time > self.thresholds['response_time']['critical']:
                self._create_alert(
                    component='rag_system',
                    metric='response_time',
                    current_value=quality_metrics.response_time,
                    threshold=self.thresholds['response_time']['critical'],
                    severity=AlertSeverity.HIGH,
                    message=f"Critical slow response time: {quality_metrics.response_time:.2f}s"
                )
                
        except Exception as e:
            self.logger.error(f"Error checking quality alerts: {e}")

    def _check_health_alerts(self, health_metrics: SystemHealthMetrics) -> None:
        """Check for health-related alerts."""
        try:
            # Check error rate
            if health_metrics.error_rate > self.thresholds['error_rate']['critical']:
                self._create_alert(
                    component=health_metrics.component_name,
                    metric='error_rate',
                    current_value=health_metrics.error_rate,
                    threshold=self.thresholds['error_rate']['critical'],
                    severity=AlertSeverity.CRITICAL,
                    message=f"Critical error rate: {health_metrics.error_rate:.2f}"
                )
            elif health_metrics.error_rate > self.thresholds['error_rate']['warning']:
                self._create_alert(
                    component=health_metrics.component_name,
                    metric='error_rate',
                    current_value=health_metrics.error_rate,
                    threshold=self.thresholds['error_rate']['warning'],
                    severity=AlertSeverity.MEDIUM,
                    message=f"High error rate: {health_metrics.error_rate:.2f}"
                )
            
            # Check component status
            if health_metrics.status == 'unhealthy':
                self._create_alert(
                    component=health_metrics.component_name,
                    metric='component_status',
                    current_value=0.0,
                    threshold=1.0,
                    severity=AlertSeverity.HIGH,
                    message=f"Component {health_metrics.component_name} is unhealthy"
                )
                
        except Exception as e:
            self.logger.error(f"Error checking health alerts: {e}")

    def _check_alert_conditions(self) -> None:
        """Check for various alert conditions."""
        try:
            # Check confidence score trends
            if len(self.confidence_score_history) >= 10:
                recent_scores = list(self.confidence_score_history)[-10:]
                avg_recent = statistics.mean(recent_scores)
                
                if avg_recent < self.thresholds['confidence_score']['warning']:
                    self._create_alert(
                        component='rag_system',
                        metric='confidence_trend',
                        current_value=avg_recent,
                        threshold=self.thresholds['confidence_score']['warning'],
                        severity=AlertSeverity.MEDIUM,
                        message=f"Declining confidence trend: {avg_recent:.2f}"
                    )
            
            # Check for system overload
            recent_quality_metrics = [
                m for m in self.response_quality_metrics 
                if m.timestamp > datetime.now() - timedelta(minutes=5)
            ]
            
            if len(recent_quality_metrics) > 50:  # More than 50 requests in 5 minutes
                avg_response_time = statistics.mean([m.response_time for m in recent_quality_metrics])
                if avg_response_time > self.thresholds['response_time']['warning']:
                    self._create_alert(
                        component='rag_system',
                        metric='system_load',
                        current_value=len(recent_quality_metrics),
                        threshold=50.0,
                        severity=AlertSeverity.MEDIUM,
                        message=f"High system load: {len(recent_quality_metrics)} requests in 5 minutes"
                    )
                    
        except Exception as e:
            self.logger.error(f"Error checking alert conditions: {e}")

    def _create_alert(self, 
                     component: str, 
                     metric: str, 
                     current_value: float,
                     threshold: float,
                     severity: AlertSeverity,
                     message: str) -> None:
        """Create and process a new alert."""
        try:
            alert_id = f"{component}_{metric}_{int(datetime.now().timestamp())}"
            
            # Check if similar alert already exists
            existing_alert_key = f"{component}_{metric}"
            if existing_alert_key in self.active_alerts:
                # Update existing alert
                existing_alert = self.active_alerts[existing_alert_key]
                existing_alert.current_value = current_value
                existing_alert.timestamp = datetime.now()
                return
            
            # Create new alert
            alert = PerformanceAlert(
                id=alert_id,
                timestamp=datetime.now(),
                severity=severity,
                component=component,
                metric=metric,
                current_value=current_value,
                threshold=threshold,
                message=message
            )
            
            # Store alert
            self.active_alerts[existing_alert_key] = alert
            self.alert_history.append(alert)
            
            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Alert callback failed: {e}")
            
            # Log alert
            self.logger.warning(f"ALERT [{severity.value.upper()}] {component}.{metric}: {message}")
            
        except Exception as e:
            self.logger.error(f"Failed to create alert: {e}")

    def resolve_alert(self, component: str, metric: str) -> bool:
        """
        Resolve an active alert.
        
        Args:
            component: Component name
            metric: Metric name
            
        Returns:
            True if alert was resolved, False otherwise
        """
        try:
            alert_key = f"{component}_{metric}"
            if alert_key in self.active_alerts:
                alert = self.active_alerts[alert_key]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                del self.active_alerts[alert_key]
                
                self.logger.info(f"Alert resolved: {component}.{metric}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to resolve alert: {e}")
            return False

    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            hours: Number of hours to include in report
            
        Returns:
            Dictionary containing comprehensive performance report
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filter recent data
            recent_quality_metrics = [
                m for m in self.response_quality_metrics 
                if m.timestamp >= cutoff_time
            ]
            
            recent_health_metrics = {}
            for component, metrics_deque in self.system_health_metrics.items():
                recent_health_metrics[component] = [
                    m for m in metrics_deque if m.timestamp >= cutoff_time
                ]
            
            # Generate report
            report = {
                'report_period': {
                    'start': cutoff_time.isoformat(),
                    'end': datetime.now().isoformat(),
                    'duration_hours': hours
                },
                'response_quality': self._generate_quality_report(recent_quality_metrics),
                'system_health': self._generate_health_report(recent_health_metrics),
                'alerts': self._generate_alerts_report(cutoff_time),
                'confidence_analysis': self.track_confidence_scores(hours * 60),
                'recommendations': self._generate_recommendations(recent_quality_metrics, recent_health_metrics)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return {'error': str(e)}

    def _generate_quality_report(self, metrics: List[ResponseQualityMetrics]) -> Dict[str, Any]:
        """Generate quality metrics report."""
        if not metrics:
            return {'status': 'no_data'}
        
        try:
            confidence_scores = [m.confidence_score for m in metrics]
            relevance_scores = [m.relevance_score for m in metrics]
            response_times = [m.response_time for m in metrics]
            
            return {
                'total_responses': len(metrics),
                'confidence': {
                    'average': statistics.mean(confidence_scores),
                    'median': statistics.median(confidence_scores),
                    'min': min(confidence_scores),
                    'max': max(confidence_scores)
                },
                'relevance': {
                    'average': statistics.mean(relevance_scores),
                    'median': statistics.median(relevance_scores)
                },
                'response_times': {
                    'average': statistics.mean(response_times),
                    'median': statistics.median(response_times),
                    'p95': sorted(response_times)[int(len(response_times) * 0.95)] if len(response_times) > 20 else max(response_times)
                },
                'quality_distribution': {
                    'high_quality': len([m for m in metrics if m.confidence_score > 0.8 and m.relevance_score > 0.8]),
                    'medium_quality': len([m for m in metrics if 0.5 <= m.confidence_score <= 0.8]),
                    'low_quality': len([m for m in metrics if m.confidence_score < 0.5])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate quality report: {e}")
            return {'error': str(e)}

    def _generate_health_report(self, health_metrics: Dict[str, List[SystemHealthMetrics]]) -> Dict[str, Any]:
        """Generate system health report."""
        try:
            component_health = {}
            
            for component, metrics in health_metrics.items():
                if not metrics:
                    component_health[component] = {'status': 'no_data'}
                    continue
                
                error_rates = [m.error_rate for m in metrics]
                response_times = [m.response_time for m in metrics]
                
                latest_status = metrics[-1].status if metrics else 'unknown'
                
                component_health[component] = {
                    'status': latest_status,
                    'metrics_count': len(metrics),
                    'error_rate': {
                        'average': statistics.mean(error_rates),
                        'max': max(error_rates)
                    },
                    'response_time': {
                        'average': statistics.mean(response_times),
                        'max': max(response_times)
                    }
                }
            
            return {
                'components': component_health,
                'overall_status': self._calculate_overall_health_status(component_health)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate health report: {e}")
            return {'error': str(e)}

    def _calculate_overall_health_status(self, component_health: Dict[str, Any]) -> str:
        """Calculate overall system health status."""
        try:
            if not component_health:
                return 'unknown'
            
            unhealthy_count = sum(1 for health in component_health.values() 
                                if health.get('status') == 'unhealthy')
            degraded_count = sum(1 for health in component_health.values() 
                               if health.get('status') == 'degraded')
            
            total_components = len(component_health)
            
            if unhealthy_count > 0:
                return 'unhealthy'
            elif degraded_count > total_components * 0.3:  # More than 30% degraded
                return 'degraded'
            else:
                return 'healthy'
                
        except Exception as e:
            self.logger.error(f"Failed to calculate overall health status: {e}")
            return 'unknown'

    def _generate_alerts_report(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Generate alerts report."""
        try:
            recent_alerts = [
                alert for alert in self.alert_history 
                if alert.timestamp >= cutoff_time
            ]
            
            return {
                'total_alerts': len(recent_alerts),
                'active_alerts': len(self.active_alerts),
                'by_severity': {
                    'critical': len([a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]),
                    'high': len([a for a in recent_alerts if a.severity == AlertSeverity.HIGH]),
                    'medium': len([a for a in recent_alerts if a.severity == AlertSeverity.MEDIUM]),
                    'low': len([a for a in recent_alerts if a.severity == AlertSeverity.LOW])
                },
                'by_component': {
                    component: len([a for a in recent_alerts if a.component == component])
                    for component in set(a.component for a in recent_alerts)
                },
                'resolution_rate': len([a for a in recent_alerts if a.resolved]) / len(recent_alerts) if recent_alerts else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate alerts report: {e}")
            return {'error': str(e)}

    def _generate_recommendations(self, 
                                quality_metrics: List[ResponseQualityMetrics],
                                health_metrics: Dict[str, List[SystemHealthMetrics]]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        try:
            # Analyze quality metrics
            if quality_metrics:
                avg_confidence = statistics.mean([m.confidence_score for m in quality_metrics])
                avg_response_time = statistics.mean([m.response_time for m in quality_metrics])
                
                if avg_confidence < 0.6:
                    recommendations.append("Consider improving FAQ content quality or expanding the knowledge base")
                
                if avg_response_time > 2.0:
                    recommendations.append("Optimize response generation performance - consider caching or model optimization")
                
                low_quality_count = len([m for m in quality_metrics if m.confidence_score < 0.5])
                if low_quality_count > len(quality_metrics) * 0.2:
                    recommendations.append("High number of low-quality responses - review query processing and FAQ matching")
            
            # Analyze health metrics
            for component, metrics in health_metrics.items():
                if not metrics:
                    continue
                
                avg_error_rate = statistics.mean([m.error_rate for m in metrics])
                if avg_error_rate > 0.05:
                    recommendations.append(f"High error rate in {component} - investigate and fix underlying issues")
            
            # Check active alerts
            if len(self.active_alerts) > 5:
                recommendations.append("Multiple active alerts - prioritize alert resolution and system stabilization")
            
            if not recommendations:
                recommendations.append("System performance is within acceptable ranges")
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to analysis error")
        
        return recommendations

    def _cleanup_old_data(self) -> None:
        """Clean up old performance data to prevent memory issues."""
        try:
            # Clean up old alert history (keep only last 30 days)
            cutoff_time = datetime.now() - timedelta(days=30)
            self.alert_history = [
                alert for alert in self.alert_history 
                if alert.timestamp >= cutoff_time
            ]
            
            # The deques have maxlen set, so they self-manage
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")

    def stop_monitoring(self) -> None:
        """Stop the performance monitoring system."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        # Save final data
        self._save_persistent_data()
        
        self.logger.info("Performance monitoring stopped")

    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get list of currently active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[PerformanceAlert]:
        """Get alert history for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history 
            if alert.timestamp >= cutoff_time
        ]

    def update_thresholds(self, new_thresholds: Dict[str, Dict[str, float]]) -> None:
        """Update performance monitoring thresholds."""
        try:
            self.thresholds.update(new_thresholds)
            self._save_persistent_data()
            self.logger.info("Performance thresholds updated")
        except Exception as e:
            self.logger.error(f"Failed to update thresholds: {e}")

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add a callback function for alert notifications."""
        self.alert_callbacks.append(callback)
        self.logger.info("Alert callback added")