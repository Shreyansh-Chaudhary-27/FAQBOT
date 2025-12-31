from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
import os
from ..interfaces.base import AnalyticsManagerInterface, ProcessedQuery, Response
from ..utils.logging import get_rag_logger


class AnalyticsManager(AnalyticsManagerInterface):
    """
    Comprehensive analytics and logging implementation for RAG system.
    
    Provides detailed query pattern analysis, performance metrics collection,
    and system event tracking with persistent storage and advanced analytics.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.logger = get_rag_logger('analytics_manager')
        
        # In-memory storage for real-time analytics
        self.query_logs: List[Dict[str, Any]] = []
        self.ingestion_logs: List[Dict[str, Any]] = []
        self.system_event_logs: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.performance_metrics = {
            'response_times': [],
            'confidence_scores': [],
            'query_processing_times': [],
            'embedding_generation_times': [],
            'vector_search_times': []
        }
        
        # Query pattern analysis
        self.query_patterns = {
            'common_intents': Counter(),
            'language_distribution': Counter(),
            'query_lengths': [],
            'typo_corrections': [],
            'failed_queries': [],
            'popular_topics': Counter()
        }
        
        # System health tracking
        self.system_health = {
            'component_errors': defaultdict(int),
            'error_patterns': Counter(),
            'uptime_events': [],
            'resource_usage': []
        }
        
        # Persistent storage setup
        self.storage_path = storage_path or "analytics_data"
        self._ensure_storage_directory()
        self._load_persistent_data()
        
        self.logger.info("Enhanced AnalyticsManager initialized with comprehensive tracking.")

    def _ensure_storage_directory(self) -> None:
        """Ensure analytics storage directory exists."""
        try:
            os.makedirs(self.storage_path, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Failed to create storage directory: {e}")
    
    def _load_persistent_data(self) -> None:
        """Load persistent analytics data from storage."""
        try:
            # Load query patterns
            patterns_file = os.path.join(self.storage_path, "query_patterns.json")
            if os.path.exists(patterns_file):
                with open(patterns_file, 'r') as f:
                    data = json.load(f)
                    self.query_patterns['common_intents'] = Counter(data.get('common_intents', {}))
                    self.query_patterns['language_distribution'] = Counter(data.get('language_distribution', {}))
                    self.query_patterns['popular_topics'] = Counter(data.get('popular_topics', {}))
            
            # Load performance metrics
            metrics_file = os.path.join(self.storage_path, "performance_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    self.performance_metrics.update(json.load(f))
            
            self.logger.info("Loaded persistent analytics data")
        except Exception as e:
            self.logger.warning(f"Failed to load persistent data: {e}")
    
    def _save_persistent_data(self) -> None:
        """Save analytics data to persistent storage."""
        try:
            # Save query patterns
            patterns_file = os.path.join(self.storage_path, "query_patterns.json")
            patterns_data = {
                'common_intents': dict(self.query_patterns['common_intents']),
                'language_distribution': dict(self.query_patterns['language_distribution']),
                'popular_topics': dict(self.query_patterns['popular_topics'])
            }
            with open(patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            # Save performance metrics (keep only recent data to prevent file bloat)
            metrics_file = os.path.join(self.storage_path, "performance_metrics.json")
            recent_metrics = {
                key: values[-1000:] if isinstance(values, list) else values
                for key, values in self.performance_metrics.items()
            }
            with open(metrics_file, 'w') as f:
                json.dump(recent_metrics, f, indent=2)
            
            self.logger.debug("Saved persistent analytics data")
        except Exception as e:
            self.logger.warning(f"Failed to save persistent data: {e}")

    def log_query(self, query_id: str, query_text: str, processed_query: ProcessedQuery, response: Response, timestamp: datetime) -> None:
        """
        Log comprehensive query information with detailed analytics tracking.
        
        Args:
            query_id: Unique identifier for the query
            query_text: Original user query text
            processed_query: Processed query object with corrections and analysis
            response: Generated response object
            timestamp: Query processing timestamp
        """
        # Create comprehensive log entry
        log_entry = {
            "query_id": query_id,
            "query_text": query_text,
            "query_length": len(query_text),
            "processed_query": {
                "corrected_query": processed_query.corrected_query,
                "intent": processed_query.intent,
                "language": processed_query.language,
                "confidence": processed_query.confidence,
                "expanded_queries_count": len(processed_query.expanded_queries),
                "typo_corrected": query_text != processed_query.corrected_query
            },
            "response": {
                "confidence": response.confidence,
                "generation_method": response.generation_method,
                "context_used": response.context_used,
                "source_faqs_count": len(response.source_faqs),
                "response_length": len(response.text)
            },
            "timestamp": timestamp.isoformat(),
            "processing_metadata": response.metadata
        }
        
        # Store in query logs
        self.query_logs.append(log_entry)
        
        # Update query patterns
        self.query_patterns['common_intents'][processed_query.intent] += 1
        self.query_patterns['language_distribution'][processed_query.language] += 1
        self.query_patterns['query_lengths'].append(len(query_text))
        
        # Track typo corrections
        if query_text != processed_query.corrected_query:
            self.query_patterns['typo_corrections'].append({
                'original': query_text,
                'corrected': processed_query.corrected_query,
                'timestamp': timestamp.isoformat()
            })
        
        # Track failed queries (low confidence responses)
        if response.confidence < 0.3:
            self.query_patterns['failed_queries'].append({
                'query': query_text,
                'confidence': response.confidence,
                'timestamp': timestamp.isoformat(),
                'reason': 'low_confidence'
            })
        
        # Extract and track topics from successful responses
        if response.confidence > 0.7 and response.source_faqs:
            for faq in response.source_faqs:
                for keyword in faq.keywords:
                    self.query_patterns['popular_topics'][keyword.lower()] += 1
        
        # Track performance metrics
        if 'response_time' in response.metadata:
            self.performance_metrics['response_times'].append(response.metadata['response_time'])
        
        self.performance_metrics['confidence_scores'].append(response.confidence)
        
        # Periodic data persistence (every 10 queries)
        if len(self.query_logs) % 10 == 0:
            self._save_persistent_data()
        
        self.logger.debug(f"Comprehensive query logged: {query_id} (confidence: {response.confidence:.2f})")

    def log_document_ingestion(self, document_path: str, faqs_ingested: int, timestamp: datetime, status: str, error: Optional[str] = None) -> None:
        """
        Log document ingestion events with detailed tracking.
        
        Args:
            document_path: Path to the ingested document
            faqs_ingested: Number of FAQs successfully extracted
            timestamp: Ingestion timestamp
            status: Ingestion status (success, failed, partial, etc.)
            error: Error message if ingestion failed
        """
        log_entry = {
            "document_path": document_path,
            "document_name": os.path.basename(document_path),
            "faqs_ingested": faqs_ingested,
            "timestamp": timestamp.isoformat(),
            "status": status,
            "error": error,
            "file_size": self._get_file_size(document_path),
            "processing_duration": None  # Could be enhanced to track processing time
        }
        
        self.ingestion_logs.append(log_entry)
        
        # Track ingestion patterns
        if status == "failed" and error:
            self.system_health['component_errors']['document_ingestion'] += 1
            self.system_health['error_patterns'][f"ingestion_error: {error[:50]}"] += 1
        
        self.logger.debug(f"Document ingestion logged: {document_path} - {status} ({faqs_ingested} FAQs)")
    
    def _get_file_size(self, file_path: str) -> Optional[int]:
        """Get file size in bytes, return None if file doesn't exist."""
        try:
            return os.path.getsize(file_path)
        except (OSError, FileNotFoundError):
            return None

    def log_system_event(self, event_type: str, details: Dict[str, Any], timestamp: datetime) -> None:
        """
        Log system events with categorization and impact tracking.
        
        Args:
            event_type: Type of system event (startup, shutdown, error, etc.)
            details: Event-specific details and metadata
            timestamp: Event timestamp
        """
        log_entry = {
            "event_type": event_type,
            "details": details,
            "timestamp": timestamp.isoformat(),
            "severity": self._determine_event_severity(event_type, details),
            "component": self._extract_component_from_event(event_type, details)
        }
        
        self.system_event_logs.append(log_entry)
        
        # Track system health patterns
        if "error" in event_type.lower() or "failed" in event_type.lower():
            component = log_entry["component"]
            self.system_health['component_errors'][component] += 1
            self.system_health['error_patterns'][event_type] += 1
        
        # Track uptime events
        if event_type in ["system_startup", "system_shutdown", "component_restart"]:
            self.system_health['uptime_events'].append({
                'event': event_type,
                'timestamp': timestamp.isoformat(),
                'details': details
            })
        
        self.logger.debug(f"System event logged: {event_type} (severity: {log_entry['severity']})")
    
    def _determine_event_severity(self, event_type: str, details: Dict[str, Any]) -> str:
        """Determine event severity based on type and details."""
        if "error" in event_type.lower() or "failed" in event_type.lower():
            return "high"
        elif "warning" in event_type.lower() or "degraded" in event_type.lower():
            return "medium"
        elif "startup" in event_type.lower() or "shutdown" in event_type.lower():
            return "medium"
        else:
            return "low"
    
    def _extract_component_from_event(self, event_type: str, details: Dict[str, Any]) -> str:
        """Extract component name from event type or details."""
        # Try to extract from event type
        for component in ["vectorizer", "query_processor", "response_generator", "vector_store", "conversation_manager"]:
            if component in event_type.lower():
                return component
        
        # Try to extract from details
        if "component" in details:
            return details["component"]
        
        return "system"

    def get_query_patterns(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Retrieve comprehensive query patterns and trends analysis.
        
        Args:
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            
        Returns:
            List of query pattern analysis results
        """
        self.logger.info("Analyzing comprehensive query patterns...")
        
        # Filter logs by date range
        filtered_logs = self._filter_logs_by_date(self.query_logs, start_date, end_date)
        
        if not filtered_logs:
            return []
        
        # Analyze query patterns
        patterns = []
        
        # 1. Intent distribution analysis
        intent_counts = Counter()
        for log in filtered_logs:
            intent_counts[log["processed_query"]["intent"]] += 1
        
        patterns.append({
            "pattern_type": "intent_distribution",
            "data": dict(intent_counts.most_common(10)),
            "total_queries": len(filtered_logs),
            "analysis_period": {
                "start": start_date.isoformat() if start_date else "beginning",
                "end": end_date.isoformat() if end_date else "now"
            }
        })
        
        # 2. Language distribution
        language_counts = Counter()
        for log in filtered_logs:
            language_counts[log["processed_query"]["language"]] += 1
        
        patterns.append({
            "pattern_type": "language_distribution",
            "data": dict(language_counts),
            "dominant_language": language_counts.most_common(1)[0] if language_counts else None
        })
        
        # 3. Query complexity analysis
        query_lengths = [log["query_length"] for log in filtered_logs]
        avg_length = sum(query_lengths) / len(query_lengths) if query_lengths else 0
        
        patterns.append({
            "pattern_type": "query_complexity",
            "data": {
                "average_length": avg_length,
                "short_queries": len([l for l in query_lengths if l < 20]),
                "medium_queries": len([l for l in query_lengths if 20 <= l < 100]),
                "long_queries": len([l for l in query_lengths if l >= 100])
            }
        })
        
        # 4. Typo correction patterns
        typo_rate = len([log for log in filtered_logs if log["processed_query"]["typo_corrected"]]) / len(filtered_logs)
        patterns.append({
            "pattern_type": "typo_correction",
            "data": {
                "typo_rate": typo_rate,
                "queries_with_typos": len([log for log in filtered_logs if log["processed_query"]["typo_corrected"]]),
                "recent_corrections": self.query_patterns['typo_corrections'][-10:]
            }
        })
        
        # 5. Response quality patterns
        confidence_scores = [log["response"]["confidence"] for log in filtered_logs]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        patterns.append({
            "pattern_type": "response_quality",
            "data": {
                "average_confidence": avg_confidence,
                "high_confidence_queries": len([c for c in confidence_scores if c > 0.8]),
                "medium_confidence_queries": len([c for c in confidence_scores if 0.5 <= c <= 0.8]),
                "low_confidence_queries": len([c for c in confidence_scores if c < 0.5])
            }
        })
        
        # 6. Popular topics analysis
        patterns.append({
            "pattern_type": "popular_topics",
            "data": dict(self.query_patterns['popular_topics'].most_common(15)),
            "trending_topics": self._get_trending_topics(filtered_logs)
        })
        
        # 7. Temporal patterns (hourly distribution)
        hourly_distribution = self._analyze_temporal_patterns(filtered_logs)
        patterns.append({
            "pattern_type": "temporal_patterns",
            "data": hourly_distribution
        })
        
        return patterns
    
    def _filter_logs_by_date(self, logs: List[Dict[str, Any]], start_date: Optional[datetime], end_date: Optional[datetime]) -> List[Dict[str, Any]]:
        """Filter logs by date range."""
        filtered = []
        for log in logs:
            log_time = datetime.fromisoformat(log["timestamp"])
            if (start_date is None or log_time >= start_date) and \
               (end_date is None or log_time <= end_date):
                filtered.append(log)
        return filtered
    
    def _get_trending_topics(self, filtered_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify trending topics from recent queries."""
        # Simple trending analysis - topics mentioned in recent high-confidence queries
        recent_topics = Counter()
        recent_cutoff = datetime.now() - timedelta(days=7)
        
        for log in filtered_logs:
            log_time = datetime.fromisoformat(log["timestamp"])
            if log_time >= recent_cutoff and log["response"]["confidence"] > 0.7:
                # Extract topics from query text (simple keyword extraction)
                query_words = log["query_text"].lower().split()
                for word in query_words:
                    if len(word) > 3:  # Filter out short words
                        recent_topics[word] += 1
        
        return [{"topic": topic, "mentions": count} for topic, count in recent_topics.most_common(10)]
    
    def _analyze_temporal_patterns(self, filtered_logs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze temporal query patterns by hour of day."""
        hourly_counts = defaultdict(int)
        for log in filtered_logs:
            log_time = datetime.fromisoformat(log["timestamp"])
            hourly_counts[log_time.hour] += 1
        return dict(hourly_counts)

    def get_performance_metrics(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Retrieve comprehensive system performance metrics.
        
        Args:
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            
        Returns:
            Dictionary containing detailed performance metrics
        """
        self.logger.info("Computing comprehensive performance metrics...")
        
        # Filter logs by date range
        filtered_query_logs = self._filter_logs_by_date(self.query_logs, start_date, end_date)
        filtered_ingestion_logs = self._filter_logs_by_date(self.ingestion_logs, start_date, end_date)
        filtered_system_logs = self._filter_logs_by_date(self.system_event_logs, start_date, end_date)
        
        # Query processing metrics
        total_queries = len(filtered_query_logs)
        successful_queries = len([log for log in filtered_query_logs if log["response"]["confidence"] > 0.5])
        high_confidence_queries = len([log for log in filtered_query_logs if log["response"]["confidence"] > 0.8])
        
        # Confidence analysis
        confidence_scores = [log["response"]["confidence"] for log in filtered_query_logs]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        min_confidence = min(confidence_scores) if confidence_scores else 0.0
        max_confidence = max(confidence_scores) if confidence_scores else 0.0
        
        # Response time analysis
        response_times = [
            log["processing_metadata"].get("response_time", 0) 
            for log in filtered_query_logs 
            if "response_time" in log.get("processing_metadata", {})
        ]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        # Generation method distribution
        generation_methods = Counter([log["response"]["generation_method"] for log in filtered_query_logs])
        
        # Context usage analysis
        context_used_count = len([log for log in filtered_query_logs if log["response"]["context_used"]])
        
        # Document ingestion metrics
        total_documents_processed = len(filtered_ingestion_logs)
        successful_ingestions = len([log for log in filtered_ingestion_logs if log["status"] == "success"])
        total_faqs_ingested = sum([log["faqs_ingested"] for log in filtered_ingestion_logs])
        
        # System health metrics
        error_events = len([log for log in filtered_system_logs if log.get("severity") == "high"])
        warning_events = len([log for log in filtered_system_logs if log.get("severity") == "medium"])
        
        # Component error analysis
        component_errors = dict(self.system_health['component_errors'])
        most_problematic_component = max(component_errors.items(), key=lambda x: x[1]) if component_errors else None
        
        # Query failure analysis
        failed_queries = [log for log in filtered_query_logs if log["response"]["confidence"] < 0.3]
        failure_reasons = Counter([
            log["processing_metadata"].get("failure_reason", "unknown") 
            for log in failed_queries
        ])
        
        # Performance trends (if we have enough data)
        performance_trend = self._calculate_performance_trend(filtered_query_logs)
        
        return {
            "analysis_period": {
                "start": start_date.isoformat() if start_date else "beginning",
                "end": end_date.isoformat() if end_date else "now",
                "duration_days": (end_date - start_date).days if start_date and end_date else None
            },
            "query_processing": {
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "high_confidence_queries": high_confidence_queries,
                "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
                "high_confidence_rate": high_confidence_queries / total_queries if total_queries > 0 else 0,
                "failed_queries": len(failed_queries),
                "failure_rate": len(failed_queries) / total_queries if total_queries > 0 else 0
            },
            "response_quality": {
                "average_confidence": avg_confidence,
                "min_confidence": min_confidence,
                "max_confidence": max_confidence,
                "confidence_distribution": {
                    "high": len([c for c in confidence_scores if c > 0.8]),
                    "medium": len([c for c in confidence_scores if 0.5 <= c <= 0.8]),
                    "low": len([c for c in confidence_scores if c < 0.5])
                }
            },
            "performance_timing": {
                "average_response_time": avg_response_time,
                "total_response_times_recorded": len(response_times),
                "performance_trend": performance_trend
            },
            "generation_methods": dict(generation_methods),
            "context_usage": {
                "queries_with_context": context_used_count,
                "context_usage_rate": context_used_count / total_queries if total_queries > 0 else 0
            },
            "document_ingestion": {
                "total_documents_processed": total_documents_processed,
                "successful_ingestions": successful_ingestions,
                "ingestion_success_rate": successful_ingestions / total_documents_processed if total_documents_processed > 0 else 0,
                "total_faqs_ingested": total_faqs_ingested,
                "average_faqs_per_document": total_faqs_ingested / successful_ingestions if successful_ingestions > 0 else 0
            },
            "system_health": {
                "error_events": error_events,
                "warning_events": warning_events,
                "total_system_events": len(filtered_system_logs),
                "most_problematic_component": most_problematic_component[0] if most_problematic_component else None,
                "component_error_counts": component_errors,
                "error_rate": error_events / len(filtered_system_logs) if filtered_system_logs else 0
            },
            "failure_analysis": {
                "failure_reasons": dict(failure_reasons),
                "common_failure_patterns": dict(self.system_health['error_patterns'].most_common(5))
            },
            "data_quality": {
                "logs_analyzed": {
                    "query_logs": len(filtered_query_logs),
                    "ingestion_logs": len(filtered_ingestion_logs),
                    "system_logs": len(filtered_system_logs)
                },
                "data_completeness": self._assess_data_completeness(filtered_query_logs)
            }
        }
    
    def _calculate_performance_trend(self, filtered_logs: List[Dict[str, Any]]) -> str:
        """Calculate performance trend over time."""
        if len(filtered_logs) < 10:
            return "insufficient_data"
        
        # Split logs into two halves and compare average confidence
        mid_point = len(filtered_logs) // 2
        first_half = filtered_logs[:mid_point]
        second_half = filtered_logs[mid_point:]
        
        first_half_confidence = sum([log["response"]["confidence"] for log in first_half]) / len(first_half)
        second_half_confidence = sum([log["response"]["confidence"] for log in second_half]) / len(second_half)
        
        if second_half_confidence > first_half_confidence + 0.05:
            return "improving"
        elif second_half_confidence < first_half_confidence - 0.05:
            return "declining"
        else:
            return "stable"
    
    def _assess_data_completeness(self, filtered_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Assess completeness of logged data."""
        if not filtered_logs:
            return {"overall": 0.0}
        
        completeness = {}
        
        # Check for response time data
        response_time_completeness = len([
            log for log in filtered_logs 
            if "response_time" in log.get("processing_metadata", {})
        ]) / len(filtered_logs)
        
        # Check for confidence scores
        confidence_completeness = len([
            log for log in filtered_logs 
            if "confidence" in log.get("response", {})
        ]) / len(filtered_logs)
        
        # Check for intent data
        intent_completeness = len([
            log for log in filtered_logs 
            if log.get("processed_query", {}).get("intent") not in [None, "", "unknown"]
        ]) / len(filtered_logs)
        
        completeness = {
            "response_times": response_time_completeness,
            "confidence_scores": confidence_completeness,
            "intent_extraction": intent_completeness,
            "overall": (response_time_completeness + confidence_completeness + intent_completeness) / 3
        }
        
        return completeness
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "component_errors": dict(self.system_health['component_errors']),
            "error_patterns": dict(self.system_health['error_patterns'].most_common(10)),
            "recent_uptime_events": self.system_health['uptime_events'][-10:],
            "total_queries_processed": len(self.query_logs),
            "total_documents_processed": len(self.ingestion_logs),
            "system_events_logged": len(self.system_event_logs),
            "data_storage_status": {
                "storage_path": self.storage_path,
                "persistent_data_available": os.path.exists(self.storage_path)
            }
        }
    
    def export_analytics_data(self, export_path: str, format: str = "json") -> bool:
        """
        Export analytics data to file.
        
        Args:
            export_path: Path to export file
            format: Export format (json, csv)
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            if format.lower() == "json":
                export_data = {
                    "query_logs": self.query_logs,
                    "ingestion_logs": self.ingestion_logs,
                    "system_event_logs": self.system_event_logs,
                    "query_patterns": {
                        "common_intents": dict(self.query_patterns['common_intents']),
                        "language_distribution": dict(self.query_patterns['language_distribution']),
                        "popular_topics": dict(self.query_patterns['popular_topics'])
                    },
                    "performance_metrics": self.performance_metrics,
                    "system_health": {
                        "component_errors": dict(self.system_health['component_errors']),
                        "error_patterns": dict(self.system_health['error_patterns'])
                    },
                    "export_timestamp": datetime.now().isoformat()
                }
                
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                self.logger.info(f"Analytics data exported to {export_path}")
                return True
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to export analytics data: {e}")
            return False