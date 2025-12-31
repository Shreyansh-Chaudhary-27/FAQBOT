"""
System Improvement Manager

Comprehensive system improvement capabilities including retraining mechanisms,
embedding update and version control, and A/B testing framework for continuous
system enhancement based on performance analytics and user feedback.
"""

import json
import os
import shutil
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import threading
import time

from ..interfaces.base import (
    FAQEntry, ProcessedQuery, Response, AnalyticsManagerInterface,
    FeedbackManagerInterface, FAQVectorizerInterface, VectorStoreInterface
)
from ..utils.logging import get_rag_logger
from ..config.settings import rag_config


class ImprovementStrategy(Enum):
    """System improvement strategies."""
    RETRAIN_EMBEDDINGS = "retrain_embeddings"
    UPDATE_FAQ_CONTENT = "update_faq_content"
    OPTIMIZE_QUERY_PROCESSING = "optimize_query_processing"
    ENHANCE_RESPONSE_GENERATION = "enhance_response_generation"
    ADJUST_SIMILARITY_THRESHOLDS = "adjust_similarity_thresholds"


class ABTestStatus(Enum):
    """A/B test status."""
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class EmbeddingVersion:
    """Data model for embedding versions."""
    version_id: str
    created_at: datetime
    model_name: str
    embedding_dimension: int
    total_faqs: int
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    backup_path: Optional[str] = None


@dataclass
class ABTestConfiguration:
    """Data model for A/B test configuration."""
    test_id: str
    name: str
    description: str
    strategy: ImprovementStrategy
    control_config: Dict[str, Any]
    treatment_config: Dict[str, Any]
    traffic_split: float  # Percentage for treatment (0.0 to 1.0)
    success_metrics: List[str]
    start_date: datetime
    end_date: datetime
    status: ABTestStatus = ABTestStatus.PLANNED
    results: Optional[Dict[str, Any]] = None


@dataclass
class ImprovementRecommendation:
    """Data model for system improvement recommendations."""
    recommendation_id: str
    strategy: ImprovementStrategy
    priority: str  # high, medium, low
    confidence: float
    description: str
    expected_impact: str
    implementation_effort: str
    supporting_data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)


class SystemImprovementManager:
    """
    Comprehensive system improvement manager that provides retraining mechanisms,
    embedding version control, and A/B testing framework for continuous enhancement.
    
    Features:
    - Automated retraining based on performance metrics
    - Embedding version control and rollback capabilities
    - A/B testing framework for system improvements
    - Performance-based improvement recommendations
    - Continuous monitoring and optimization
    """

    def __init__(self,
                 analytics_manager: Optional[AnalyticsManagerInterface] = None,
                 feedback_manager: Optional[FeedbackManagerInterface] = None,
                 vectorizer: Optional[FAQVectorizerInterface] = None,
                 vector_store: Optional[VectorStoreInterface] = None,
                 storage_path: Optional[str] = None):
        """
        Initialize the system improvement manager.
        
        Args:
            analytics_manager: Analytics manager for performance data
            feedback_manager: Feedback manager for user feedback data
            vectorizer: FAQ vectorizer for embedding operations
            vector_store: Vector store for embedding management
            storage_path: Path for storing improvement data
        """
        self.logger = get_rag_logger('system_improvement')
        self.analytics_manager = analytics_manager
        self.feedback_manager = feedback_manager
        self.vectorizer = vectorizer
        self.vector_store = vector_store
        
        # Storage configuration
        self.storage_path = storage_path or "system_improvement_data"
        self.embeddings_backup_path = os.path.join(self.storage_path, "embedding_versions")
        self.ab_tests_path = os.path.join(self.storage_path, "ab_tests")
        
        # Embedding version control
        self.embedding_versions: List[EmbeddingVersion] = []
        self.current_embedding_version: Optional[str] = None
        
        # A/B testing framework
        self.active_ab_tests: Dict[str, ABTestConfiguration] = {}
        self.ab_test_history: List[ABTestConfiguration] = []
        self.ab_test_assignments: Dict[str, str] = {}  # user_id -> test_id
        
        # Improvement recommendations
        self.improvement_recommendations: List[ImprovementRecommendation] = []
        self.improvement_history: List[Dict[str, Any]] = []
        
        # Retraining configuration
        self.retraining_config = {
            'auto_retrain_enabled': True,
            'confidence_threshold': 0.6,  # Retrain if avg confidence drops below this
            'feedback_threshold': 3.0,    # Retrain if avg feedback drops below this
            'min_data_points': 100,       # Minimum queries before considering retraining
            'retrain_interval_days': 7,   # Check for retraining every N days
            'performance_decline_threshold': 0.1  # Retrain if performance drops by this much
        }
        
        # Monitoring and automation
        self.monitoring_active = True
        self.monitoring_thread = None
        
        # Initialize storage and load data
        self._ensure_storage_directories()
        self._load_persistent_data()
        self._start_monitoring_thread()
        
        self.logger.info("System Improvement Manager initialized with comprehensive capabilities")

    def analyze_and_adapt(self) -> None:
        """
        Analyzes feedback and performance metrics to trigger system adaptations.
        This method can be called on-demand to initiate an improvement cycle.
        """
        self.logger.info("Initiating on-demand analysis and adaptation cycle...")
        try:
            # Check for retraining opportunities
            if self.retraining_config['auto_retrain_enabled']:
                self._check_retraining_conditions()
            
            # Update improvement recommendations
            self._update_improvement_recommendations()
            
            # Monitor A/B tests (if any are active)
            self._monitor_ab_tests()
            
            # Clean up old data
            self._cleanup_old_data()
            
            # Save data after adaptation
            self._save_persistent_data()
            
            self.logger.info("On-demand analysis and adaptation cycle completed.")
        except Exception as e:
            self.logger.error(f"Error during on-demand analysis and adaptation: {e}")

    def _ensure_storage_directories(self) -> None:
        """Ensure all storage directories exist."""
        try:
            os.makedirs(self.storage_path, exist_ok=True)
            os.makedirs(self.embeddings_backup_path, exist_ok=True)
            os.makedirs(self.ab_tests_path, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Failed to create storage directories: {e}")

    def _load_persistent_data(self) -> None:
        """Load persistent improvement data from storage."""
        try:
            # Load embedding versions
            versions_file = os.path.join(self.storage_path, "embedding_versions.json")
            if os.path.exists(versions_file):
                with open(versions_file, 'r') as f:
                    versions_data = json.load(f)
                    for version_data in versions_data:
                        version = EmbeddingVersion(
                            version_id=version_data['version_id'],
                            created_at=datetime.fromisoformat(version_data['created_at']),
                            model_name=version_data['model_name'],
                            embedding_dimension=version_data['embedding_dimension'],
                            total_faqs=version_data['total_faqs'],
                            performance_metrics=version_data['performance_metrics'],
                            metadata=version_data.get('metadata', {}),
                            backup_path=version_data.get('backup_path')
                        )
                        self.embedding_versions.append(version)
            
            # Load current version
            current_version_file = os.path.join(self.storage_path, "current_version.txt")
            if os.path.exists(current_version_file):
                with open(current_version_file, 'r') as f:
                    self.current_embedding_version = f.read().strip()
            
            # Load A/B tests
            ab_tests_file = os.path.join(self.storage_path, "ab_tests.json")
            if os.path.exists(ab_tests_file):
                with open(ab_tests_file, 'r') as f:
                    ab_tests_data = json.load(f)
                    for test_data in ab_tests_data:
                        test_config = ABTestConfiguration(
                            test_id=test_data['test_id'],
                            name=test_data['name'],
                            description=test_data['description'],
                            strategy=ImprovementStrategy(test_data['strategy']),
                            control_config=test_data['control_config'],
                            treatment_config=test_data['treatment_config'],
                            traffic_split=test_data['traffic_split'],
                            success_metrics=test_data['success_metrics'],
                            start_date=datetime.fromisoformat(test_data['start_date']),
                            end_date=datetime.fromisoformat(test_data['end_date']),
                            status=ABTestStatus(test_data['status']),
                            results=test_data.get('results')
                        )
                        
                        if test_config.status == ABTestStatus.RUNNING:
                            self.active_ab_tests[test_config.test_id] = test_config
                        else:
                            self.ab_test_history.append(test_config)
            
            # Load improvement recommendations
            recommendations_file = os.path.join(self.storage_path, "recommendations.json")
            if os.path.exists(recommendations_file):
                with open(recommendations_file, 'r') as f:
                    recommendations_data = json.load(f)
                    for rec_data in recommendations_data:
                        recommendation = ImprovementRecommendation(
                            recommendation_id=rec_data['recommendation_id'],
                            strategy=ImprovementStrategy(rec_data['strategy']),
                            priority=rec_data['priority'],
                            confidence=rec_data['confidence'],
                            description=rec_data['description'],
                            expected_impact=rec_data['expected_impact'],
                            implementation_effort=rec_data['implementation_effort'],
                            supporting_data=rec_data['supporting_data'],
                            created_at=datetime.fromisoformat(rec_data['created_at'])
                        )
                        self.improvement_recommendations.append(recommendation)
            
            # Load retraining configuration
            config_file = os.path.join(self.storage_path, "retraining_config.json")
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    self.retraining_config.update(json.load(f))
            
            self.logger.info("Loaded persistent improvement data")
        except Exception as e:
            self.logger.warning(f"Failed to load persistent improvement data: {e}")

    def _save_persistent_data(self) -> None:
        """Save improvement data to persistent storage."""
        try:
            # Save embedding versions
            versions_file = os.path.join(self.storage_path, "embedding_versions.json")
            versions_data = []
            for version in self.embedding_versions:
                version_data = {
                    'version_id': version.version_id,
                    'created_at': version.created_at.isoformat(),
                    'model_name': version.model_name,
                    'embedding_dimension': version.embedding_dimension,
                    'total_faqs': version.total_faqs,
                    'performance_metrics': version.performance_metrics,
                    'metadata': version.metadata,
                    'backup_path': version.backup_path
                }
                versions_data.append(version_data)
            
            with open(versions_file, 'w') as f:
                json.dump(versions_data, f, indent=2)
            
            # Save current version
            if self.current_embedding_version:
                current_version_file = os.path.join(self.storage_path, "current_version.txt")
                with open(current_version_file, 'w') as f:
                    f.write(self.current_embedding_version)
            
            # Save A/B tests
            ab_tests_file = os.path.join(self.storage_path, "ab_tests.json")
            all_tests = list(self.active_ab_tests.values()) + self.ab_test_history
            ab_tests_data = []
            for test in all_tests:
                test_data = {
                    'test_id': test.test_id,
                    'name': test.name,
                    'description': test.description,
                    'strategy': test.strategy.value,
                    'control_config': test.control_config,
                    'treatment_config': test.treatment_config,
                    'traffic_split': test.traffic_split,
                    'success_metrics': test.success_metrics,
                    'start_date': test.start_date.isoformat(),
                    'end_date': test.end_date.isoformat(),
                    'status': test.status.value,
                    'results': test.results
                }
                ab_tests_data.append(test_data)
            
            with open(ab_tests_file, 'w') as f:
                json.dump(ab_tests_data, f, indent=2)
            
            # Save improvement recommendations
            recommendations_file = os.path.join(self.storage_path, "recommendations.json")
            recommendations_data = []
            for rec in self.improvement_recommendations:
                rec_data = {
                    'recommendation_id': rec.recommendation_id,
                    'strategy': rec.strategy.value,
                    'priority': rec.priority,
                    'confidence': rec.confidence,
                    'description': rec.description,
                    'expected_impact': rec.expected_impact,
                    'implementation_effort': rec.implementation_effort,
                    'supporting_data': rec.supporting_data,
                    'created_at': rec.created_at.isoformat()
                }
                recommendations_data.append(rec_data)
            
            with open(recommendations_file, 'w') as f:
                json.dump(recommendations_data, f, indent=2)
            
            # Save retraining configuration
            config_file = os.path.join(self.storage_path, "retraining_config.json")
            with open(config_file, 'w') as f:
                json.dump(self.retraining_config, f, indent=2)
            
            self.logger.debug("Saved persistent improvement data")
        except Exception as e:
            self.logger.warning(f"Failed to save persistent improvement data: {e}")

    def _start_monitoring_thread(self) -> None:
        """Start background monitoring thread for automated improvements."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("Started system improvement monitoring thread")

    def _monitoring_loop(self) -> None:
        """Background monitoring loop for automated system improvements."""
        while self.monitoring_active:
            try:
                # Check for retraining opportunities
                if self.retraining_config['auto_retrain_enabled']:
                    self._check_retraining_conditions()
                
                # Update improvement recommendations
                self._update_improvement_recommendations()
                
                # Monitor A/B tests
                self._monitor_ab_tests()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Save data periodically
                self._save_persistent_data()
                
                # Sleep for monitoring interval
                time.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Error in improvement monitoring loop: {e}")
                time.sleep(1800)  # Wait 30 minutes on error

    def create_embedding_version(self, 
                               model_name: str,
                               faqs: List[FAQEntry],
                               performance_metrics: Optional[Dict[str, float]] = None) -> str:
        """
        Create a new embedding version with backup and version control.
        
        Args:
            model_name: Name of the embedding model used
            faqs: List of FAQ entries to create embeddings for
            performance_metrics: Optional performance metrics for this version
            
        Returns:
            Version ID of the created embedding version
        """
        try:
            # Generate version ID
            timestamp = datetime.now()
            version_id = f"v_{timestamp.strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(model_name.encode()).hexdigest()[:8]}"
            
            # Create backup of current embeddings if vector store available
            backup_path = None
            if self.vector_store:
                backup_path = os.path.join(self.embeddings_backup_path, f"{version_id}_backup.json")
                try:
                    current_backup = self.vector_store.backup_store(backup_path)
                    backup_path = current_backup
                    self.logger.info(f"Created embedding backup at {backup_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to create embedding backup: {e}")
            
            # Generate embeddings for FAQs if vectorizer available
            embedding_dimension = 0
            if self.vectorizer:
                try:
                    # Generate embeddings for a sample FAQ to get dimension
                    if faqs:
                        sample_embedding = self.vectorizer.generate_embeddings(faqs[0].question)
                        embedding_dimension = len(sample_embedding)
                        
                        # Vectorize all FAQs
                        vectorized_faqs = []
                        for faq in faqs:
                            vectorized_faq = self.vectorizer.vectorize_faq_entry(faq)
                            vectorized_faqs.append(vectorized_faq)
                        
                        # Store in vector store
                        if self.vector_store:
                            self.vector_store.store_vectors(vectorized_faqs, document_id=f"version_{version_id}")
                        
                        self.logger.info(f"Generated embeddings for {len(vectorized_faqs)} FAQs")
                except Exception as e:
                    self.logger.error(f"Failed to generate embeddings: {e}")
            
            # Create version record
            version = EmbeddingVersion(
                version_id=version_id,
                created_at=timestamp,
                model_name=model_name,
                embedding_dimension=embedding_dimension,
                total_faqs=len(faqs),
                performance_metrics=performance_metrics or {},
                metadata={
                    'creation_method': 'manual',
                    'faq_sources': list(set(faq.source_document for faq in faqs))
                },
                backup_path=backup_path
            )
            
            # Store version
            self.embedding_versions.append(version)
            self.current_embedding_version = version_id
            
            # Save data
            self._save_persistent_data()
            
            self.logger.info(f"Created embedding version {version_id} with {len(faqs)} FAQs")
            return version_id
            
        except Exception as e:
            self.logger.error(f"Failed to create embedding version: {e}")
            raise

    def rollback_embedding_version(self, version_id: str) -> bool:
        """
        Rollback to a previous embedding version.
        
        Args:
            version_id: ID of the version to rollback to
            
        Returns:
            True if rollback successful, False otherwise
        """
        try:
            # Find the version
            target_version = None
            for version in self.embedding_versions:
                if version.version_id == version_id:
                    target_version = version
                    break
            
            if not target_version:
                self.logger.error(f"Version {version_id} not found")
                return False
            
            # Check if backup exists
            if not target_version.backup_path or not os.path.exists(target_version.backup_path):
                self.logger.error(f"Backup for version {version_id} not found")
                return False
            
            # Restore from backup
            if self.vector_store:
                try:
                    success = self.vector_store.restore_from_backup(target_version.backup_path)
                    if not success:
                        self.logger.error(f"Failed to restore from backup {target_version.backup_path}")
                        return False
                except Exception as e:
                    self.logger.error(f"Error during backup restoration: {e}")
                    return False
            
            # Update current version
            self.current_embedding_version = version_id
            
            # Log rollback
            rollback_record = {
                'timestamp': datetime.now().isoformat(),
                'action': 'rollback',
                'from_version': self.current_embedding_version,
                'to_version': version_id,
                'reason': 'manual_rollback'
            }
            self.improvement_history.append(rollback_record)
            
            # Save data
            self._save_persistent_data()
            
            self.logger.info(f"Successfully rolled back to embedding version {version_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rollback to version {version_id}: {e}")
            return False

    def get_embedding_versions(self) -> List[Dict[str, Any]]:
        """
        Get list of all embedding versions with their metadata.
        
        Returns:
            List of embedding version information
        """
        try:
            versions_info = []
            for version in sorted(self.embedding_versions, key=lambda v: v.created_at, reverse=True):
                version_info = {
                    'version_id': version.version_id,
                    'created_at': version.created_at.isoformat(),
                    'model_name': version.model_name,
                    'embedding_dimension': version.embedding_dimension,
                    'total_faqs': version.total_faqs,
                    'performance_metrics': version.performance_metrics,
                    'is_current': version.version_id == self.current_embedding_version,
                    'has_backup': version.backup_path is not None and os.path.exists(version.backup_path) if version.backup_path else False,
                    'metadata': version.metadata
                }
                versions_info.append(version_info)
            
            return versions_info
            
        except Exception as e:
            self.logger.error(f"Failed to get embedding versions: {e}")
            return []

    def create_ab_test(self,
                      name: str,
                      description: str,
                      strategy: ImprovementStrategy,
                      control_config: Dict[str, Any],
                      treatment_config: Dict[str, Any],
                      traffic_split: float = 0.5,
                      duration_days: int = 7,
                      success_metrics: Optional[List[str]] = None) -> str:
        """
        Create a new A/B test for system improvements.
        
        Args:
            name: Name of the A/B test
            description: Description of what is being tested
            strategy: Improvement strategy being tested
            control_config: Configuration for control group
            treatment_config: Configuration for treatment group
            traffic_split: Percentage of traffic for treatment (0.0 to 1.0)
            duration_days: Duration of the test in days
            success_metrics: List of metrics to measure success
            
        Returns:
            Test ID of the created A/B test
        """
        try:
            # Generate test ID
            timestamp = datetime.now()
            test_id = f"ab_{timestamp.strftime('%Y%m%d_%H%M%S')}_{strategy.value[:10]}"
            
            # Create test configuration
            ab_test = ABTestConfiguration(
                test_id=test_id,
                name=name,
                description=description,
                strategy=strategy,
                control_config=control_config,
                treatment_config=treatment_config,
                traffic_split=traffic_split,
                success_metrics=success_metrics or ['confidence_score', 'response_time', 'user_satisfaction'],
                start_date=timestamp,
                end_date=timestamp + timedelta(days=duration_days),
                status=ABTestStatus.PLANNED
            )
            
            # Store test
            self.active_ab_tests[test_id] = ab_test
            
            # Save data
            self._save_persistent_data()
            
            self.logger.info(f"Created A/B test {test_id}: {name}")
            return test_id
            
        except Exception as e:
            self.logger.error(f"Failed to create A/B test: {e}")
            raise

    def start_ab_test(self, test_id: str) -> bool:
        """
        Start an A/B test.
        
        Args:
            test_id: ID of the test to start
            
        Returns:
            True if test started successfully, False otherwise
        """
        try:
            if test_id not in self.active_ab_tests:
                self.logger.error(f"A/B test {test_id} not found")
                return False
            
            test = self.active_ab_tests[test_id]
            if test.status != ABTestStatus.PLANNED:
                self.logger.error(f"A/B test {test_id} is not in planned status")
                return False
            
            # Update test status
            test.status = ABTestStatus.RUNNING
            test.start_date = datetime.now()
            
            # Initialize test tracking
            test_dir = os.path.join(self.ab_tests_path, test_id)
            os.makedirs(test_dir, exist_ok=True)
            
            # Save test data
            self._save_persistent_data()
            
            self.logger.info(f"Started A/B test {test_id}: {test.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start A/B test {test_id}: {e}")
            return False

    def assign_user_to_ab_test(self, user_id: str, test_id: str) -> str:
        """
        Assign a user to an A/B test group.
        
        Args:
            user_id: ID of the user
            test_id: ID of the A/B test
            
        Returns:
            Group assignment ('control' or 'treatment')
        """
        try:
            if test_id not in self.active_ab_tests:
                return 'control'  # Default to control if test not found
            
            test = self.active_ab_tests[test_id]
            if test.status != ABTestStatus.RUNNING:
                return 'control'  # Default to control if test not running
            
            # Check if user already assigned
            assignment_key = f"{user_id}_{test_id}"
            if assignment_key in self.ab_test_assignments:
                return self.ab_test_assignments[assignment_key]
            
            # Assign user based on hash and traffic split
            user_hash = hashlib.md5(f"{user_id}_{test_id}".encode()).hexdigest()
            hash_value = int(user_hash[:8], 16) / (16**8)  # Convert to 0-1 range
            
            assignment = 'treatment' if hash_value < test.traffic_split else 'control'
            self.ab_test_assignments[assignment_key] = assignment
            
            return assignment
            
        except Exception as e:
            self.logger.error(f"Failed to assign user {user_id} to A/B test {test_id}: {e}")
            return 'control'

    def record_ab_test_result(self, 
                            test_id: str, 
                            user_id: str, 
                            group: str,
                            metrics: Dict[str, float]) -> None:
        """
        Record A/B test result for analysis.
        
        Args:
            test_id: ID of the A/B test
            user_id: ID of the user
            group: Group assignment ('control' or 'treatment')
            metrics: Dictionary of metric values
        """
        try:
            if test_id not in self.active_ab_tests:
                return
            
            # Store result
            test_dir = os.path.join(self.ab_tests_path, test_id)
            results_file = os.path.join(test_dir, f"{group}_results.json")
            
            result_entry = {
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            }
            
            # Load existing results
            results = []
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
            
            # Add new result
            results.append(result_entry)
            
            # Save results
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.debug(f"Recorded A/B test result for {test_id}, user {user_id}, group {group}")
            
        except Exception as e:
            self.logger.error(f"Failed to record A/B test result: {e}")

    def analyze_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """
        Analyze A/B test results and determine statistical significance.
        
        Args:
            test_id: ID of the A/B test to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            if test_id not in self.active_ab_tests and test_id not in [t.test_id for t in self.ab_test_history]:
                return {'error': 'Test not found'}
            
            test_dir = os.path.join(self.ab_tests_path, test_id)
            if not os.path.exists(test_dir):
                return {'error': 'Test results not found'}
            
            # Load results
            control_results = []
            treatment_results = []
            
            control_file = os.path.join(test_dir, "control_results.json")
            treatment_file = os.path.join(test_dir, "treatment_results.json")
            
            if os.path.exists(control_file):
                with open(control_file, 'r') as f:
                    control_results = json.load(f)
            
            if os.path.exists(treatment_file):
                with open(treatment_file, 'r') as f:
                    treatment_results = json.load(f)
            
            if not control_results or not treatment_results:
                return {'error': 'Insufficient data for analysis'}
            
            # Analyze metrics
            analysis = {
                'test_id': test_id,
                'control_group': {
                    'sample_size': len(control_results),
                    'metrics': self._calculate_group_metrics(control_results)
                },
                'treatment_group': {
                    'sample_size': len(treatment_results),
                    'metrics': self._calculate_group_metrics(treatment_results)
                },
                'comparison': {},
                'recommendation': 'insufficient_data'
            }
            
            # Compare metrics
            for metric in ['confidence_score', 'response_time', 'user_satisfaction']:
                control_values = [r['metrics'].get(metric, 0) for r in control_results if metric in r['metrics']]
                treatment_values = [r['metrics'].get(metric, 0) for r in treatment_results if metric in r['metrics']]
                
                if control_values and treatment_values:
                    control_avg = sum(control_values) / len(control_values)
                    treatment_avg = sum(treatment_values) / len(treatment_values)
                    
                    improvement = ((treatment_avg - control_avg) / control_avg) * 100 if control_avg > 0 else 0
                    
                    analysis['comparison'][metric] = {
                        'control_average': control_avg,
                        'treatment_average': treatment_avg,
                        'improvement_percentage': improvement,
                        'is_significant': abs(improvement) > 5.0  # Simple significance test
                    }
            
            # Generate recommendation
            significant_improvements = [
                metric for metric, data in analysis['comparison'].items()
                if data.get('is_significant', False) and data.get('improvement_percentage', 0) > 0
            ]
            
            if len(significant_improvements) >= 2:
                analysis['recommendation'] = 'implement_treatment'
            elif len(significant_improvements) == 1:
                analysis['recommendation'] = 'continue_testing'
            else:
                analysis['recommendation'] = 'keep_control'
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze A/B test results for {test_id}: {e}")
            return {'error': str(e)}

    def _calculate_group_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate metrics for a test group."""
        try:
            if not results:
                return {}
            
            metrics = {}
            metric_values = defaultdict(list)
            
            # Collect all metric values
            for result in results:
                for metric, value in result.get('metrics', {}).items():
                    metric_values[metric].append(value)
            
            # Calculate averages
            for metric, values in metric_values.items():
                if values:
                    metrics[f"{metric}_average"] = sum(values) / len(values)
                    metrics[f"{metric}_count"] = len(values)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate group metrics: {e}")
            return {}

    def complete_ab_test(self, test_id: str, implement_treatment: bool = False) -> bool:
        """
        Complete an A/B test and optionally implement the treatment.
        
        Args:
            test_id: ID of the A/B test to complete
            implement_treatment: Whether to implement the treatment configuration
            
        Returns:
            True if test completed successfully, False otherwise
        """
        try:
            if test_id not in self.active_ab_tests:
                self.logger.error(f"Active A/B test {test_id} not found")
                return False
            
            test = self.active_ab_tests[test_id]
            
            # Analyze final results
            final_results = self.analyze_ab_test_results(test_id)
            
            # Update test status and results
            test.status = ABTestStatus.COMPLETED
            test.end_date = datetime.now()
            test.results = final_results
            
            # Move to history
            self.ab_test_history.append(test)
            del self.active_ab_tests[test_id]
            
            # Implement treatment if requested and recommended
            if implement_treatment and final_results.get('recommendation') == 'implement_treatment':
                self._implement_ab_test_treatment(test)
            
            # Save data
            self._save_persistent_data()
            
            self.logger.info(f"Completed A/B test {test_id}, implement_treatment={implement_treatment}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to complete A/B test {test_id}: {e}")
            return False

    def _implement_ab_test_treatment(self, test: ABTestConfiguration) -> None:
        """Implement the treatment configuration from an A/B test."""
        try:
            implementation_record = {
                'timestamp': datetime.now().isoformat(),
                'action': 'implement_ab_test_treatment',
                'test_id': test.test_id,
                'strategy': test.strategy.value,
                'treatment_config': test.treatment_config,
                'results': test.results
            }
            
            self.improvement_history.append(implementation_record)
            
            # Strategy-specific implementation
            if test.strategy == ImprovementStrategy.ADJUST_SIMILARITY_THRESHOLDS:
                # This would update system configuration
                self.logger.info(f"Implementing similarity threshold adjustment from test {test.test_id}")
            
            elif test.strategy == ImprovementStrategy.RETRAIN_EMBEDDINGS:
                # This would trigger embedding retraining
                self.logger.info(f"Implementing embedding retraining from test {test.test_id}")
            
            # Add more strategy implementations as needed
            
            self.logger.info(f"Implemented treatment from A/B test {test.test_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to implement A/B test treatment: {e}")

    def _adapt_lm_based_on_feedback(self, feedback_analysis: dict) -> None:
        """
        Adapts the Language Model's behavior based on negative feedback.
        This can include prompt engineering, adjusting re-ranking strategies, etc.
        """
        self.logger.info(f"Adapting LM based on feedback: {feedback_analysis.get('overall_statistics', {})}")

        # Example: Adjust prompt engineering based on common negative feedback themes
        # This is a placeholder for actual prompt modification logic
        negative_themes = feedback_analysis.get('negative_themes', [])
        positive_themes = feedback_analysis.get('positive_themes', [])
        if "irrelevant_response" in negative_themes:
            self.logger.info("Detected 'irrelevant_response' theme. Considering prompt adjustments for better relevance.")
            # self._update_lm_prompt_for_relevance() # Placeholder for actual implementation

        if "poor_ranking" in negative_themes:
            self.logger.info("Detected 'poor_ranking' theme. Considering adjustments to document re-ranking strategy.")
            # self._update_reranking_strategy() # Placeholder for actual implementation

        # Example: Trigger a micro-retraining or prompt update for specific scenarios
        # This would involve interacting with the RAGSystem's LM component
        self.logger.info("LM adaptation process initiated. Further specific adaptations would go here.")
        
        # After adaptation, save any changes to persistent data if necessary
        self._save_persistent_data()

    def _check_retraining_conditions(self) -> None:
        """Check if system retraining is needed based on performance metrics."""
        try:
            if not self.analytics_manager:
                return
            
            # Get recent performance metrics
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.retraining_config['retrain_interval_days'])
            
            performance_metrics = self.analytics_manager.get_performance_metrics(start_date, end_date)
            
            # Check if we have enough data
            total_queries = performance_metrics.get('query_processing', {}).get('total_queries', 0)
            if total_queries < self.retraining_config['min_data_points']:
                return
            
            # Check confidence threshold
            avg_confidence = performance_metrics.get('response_quality', {}).get('average_confidence', 1.0)
            if avg_confidence < self.retraining_config['confidence_threshold']:
                self._trigger_retraining('low_confidence', {
                    'current_confidence': avg_confidence,
                    'threshold': self.retraining_config['confidence_threshold']
                })
                return
            
            # Check feedback threshold if feedback manager available
            if self.feedback_manager:
                try:
                    feedback_analysis = self.feedback_manager.analyze_feedback(start_date, end_date)
                    avg_rating = feedback_analysis.get('overall_statistics', {}).get('average_rating', 5.0)
                    
                    if avg_rating < self.retraining_config['feedback_threshold']:
                        self.logger.warning(f"Average feedback rating ({avg_rating:.2f}) below threshold ({self.retraining_config['feedback_threshold']}).")
                        
                        # Trigger LM adaptation first
                        self._adapt_lm_based_on_feedback(feedback_analysis)

                        # Then consider full retraining if adaptation isn\'t enough or for broader issues
                        self._trigger_retraining('low_feedback', {
                            'current_rating': avg_rating,
                            'threshold': self.retraining_config['feedback_threshold']
                        })
                        return # Exit after triggering adaptation and potential retraining
                except Exception as e:
                    self.logger.warning(f"Failed to check feedback for retraining: {e}")
            
            # Check performance decline
            if len(self.embedding_versions) > 1:
                current_version = self._get_current_version()
                if current_version and current_version.performance_metrics:
                    current_performance = current_version.performance_metrics.get('confidence_score', 0)
                    recent_performance = avg_confidence
                    
                    decline = current_performance - recent_performance
                    if decline > self.retraining_config['performance_decline_threshold']:
                        self._trigger_retraining('performance_decline', {
                            'baseline_performance': current_performance,
                            'recent_performance': recent_performance,
                            'decline': decline
                        })
            
        except Exception as e:
            self.logger.error(f"Error checking retraining conditions: {e}")

    def _trigger_retraining(self, reason: str, data: Dict[str, Any]) -> None:
        """Trigger system retraining based on performance conditions."""
        try:
            self.logger.info(f"Triggering retraining due to {reason}: {data}")
            
            # Create retraining record
            retraining_record = {
                'timestamp': datetime.now().isoformat(),
                'action': 'automatic_retraining_triggered',
                'reason': reason,
                'trigger_data': data,
                'status': 'initiated'
            }
            
            self.improvement_history.append(retraining_record)
            
            # Create improvement recommendation
            recommendation = ImprovementRecommendation(
                recommendation_id=f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                strategy=ImprovementStrategy.RETRAIN_EMBEDDINGS,
                priority='high',
                confidence=0.8,
                description=f"Automatic retraining recommended due to {reason}",
                expected_impact='Improved response quality and user satisfaction',
                implementation_effort='medium',
                supporting_data=data
            )
            
            self.improvement_recommendations.append(recommendation)
            
            # Note: Actual retraining would be triggered here
            # This would involve calling the embedding generation process
            
            self.logger.info(f"Retraining recommendation created: {recommendation.recommendation_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to trigger retraining: {e}")

    def _adapt_lm_based_on_feedback(self, feedback_analysis: dict) -> None:
        """
        Adapts the Language Model's behavior based on negative feedback.
        This can include prompt engineering, adjusting re-ranking strategies, etc.
        """
        self.logger.info(f"Adapting LM based on feedback: {feedback_analysis.get('overall_statistics', {})}")

        # Example: Adjust prompt engineering based on common negative feedback themes
        negative_themes = feedback_analysis.get('negative_themes', [])
        
        # Initialize updates for RAGConfigManager
        config_updates = {}

        if "irrelevant_response" in negative_themes:
            self.logger.info("Detected 'irrelevant_response' theme. Considering prompt adjustments for better relevance.")
            # Example: Lower response temperature for more focused answers
            config_updates['response_temperature'] = max(0.1, rag_config.get_response_config().get('response_temperature', 0.7) - 0.1)
            # Example: Increase top_k for retrieval to get more diverse initial results
            config_updates['top_k_retrieval'] = rag_config.get_retrieval_config().get('top_k_retrieval', 5) + 1

        if "poor_ranking" in negative_themes:
            self.logger.info("Detected 'poor_ranking' theme. Considering adjustments to document re-ranking strategy.")
            # Example: Adjust re-ranking weight or strategy
            config_updates['reranking_strategy'] = 'semantic_reranking_with_feedback' # A hypothetical new strategy
            config_updates['reranking_weight'] = min(1.0, rag_config.get_retrieval_config().get('reranking_weight', 0.5) + 0.1)

        if "too_short" in negative_themes:
            self.logger.info("Detected 'too_short' theme. Increasing max response length.")
            config_updates['max_response_length'] = rag_config.get_response_config().get('max_response_length', 200) + 50

        if "too_long" in negative_themes:
            self.logger.info("Detected 'too_long' theme. Decreasing max response length.")
            config_updates['max_response_length'] = max(50, rag_config.get_response_config().get('max_response_length', 200) - 50)

        if "low_confidence_response" in negative_themes:
            self.logger.info("Detected 'low_confidence_response' theme. Increasing confidence threshold.")
            config_updates['confidence_threshold'] = min(0.95, rag_config.get_response_config().get('confidence_threshold', 0.6) + 0.05)

        if "inaccurate_response" in negative_themes:
            self.logger.info("Detected 'inaccurate_response' theme. Adjusting similarity threshold and context window size.")
            # Make retrieval stricter and provide more context
            config_updates['similarity_threshold'] = min(0.95, rag_config.get_vector_config().get('similarity_threshold', 0.7) + 0.05)
            config_updates['context_window_size'] = min(10, rag_config.get_response_config().get('context_window_size', 5) + 1)

        if "irrelevant_context" in negative_themes:
            self.logger.info("Detected 'irrelevant_context' theme. Adjusting similarity threshold and max results.")
            # Make retrieval stricter and fetch fewer, more relevant results
            config_updates['similarity_threshold'] = min(0.95, rag_config.get_vector_config().get('similarity_threshold', 0.7) + 0.05)
            config_updates['max_results'] = max(1, rag_config.get_vector_config().get('max_results', 10) - 1)

        if "helpful_response" in positive_themes:
            self.logger.info("Detected 'helpful_response' theme. Slightly increasing max response length and decreasing confidence threshold.")
            # Encourage slightly more comprehensive and less conservative responses
            config_updates['max_response_length'] = min(1000, rag_config.get_response_config().get('max_response_length', 500) + 25)
            config_updates['confidence_threshold'] = max(0.1, rag_config.get_response_config().get('confidence_threshold', 0.6) - 0.02)

        # Apply updates if any
        if config_updates:
            self.logger.info(f"Applying RAG config updates: {config_updates}")
            rag_config.update_config(config_updates)
        else:
            self.logger.info("No specific LM adaptation needed based on current feedback analysis.")
        
        # After adaptation, save any changes to persistent data if necessary
        self._save_persistent_data()

    def _get_current_version(self) -> Optional[EmbeddingVersion]:
        """Get the current embedding version."""
        if not self.current_embedding_version:
            return None
        
        for version in self.embedding_versions:
            if version.version_id == self.current_embedding_version:
                return version
        
        return None

    def _update_improvement_recommendations(self) -> None:
        """Update improvement recommendations based on current system state."""
        try:
            # This would analyze current system performance and generate recommendations
            # For now, we'll implement basic recommendation logic
            
            if not self.analytics_manager:
                return
            
            # Get recent performance data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            performance_metrics = self.analytics_manager.get_performance_metrics(start_date, end_date)
            
            # Check for improvement opportunities
            avg_confidence = performance_metrics.get('response_quality', {}).get('average_confidence', 1.0)
            avg_response_time = performance_metrics.get('performance_timing', {}).get('average_response_time', 0.0)
            
            # Generate recommendations based on thresholds
            if avg_confidence < 0.7 and not self._has_recent_recommendation(ImprovementStrategy.RETRAIN_EMBEDDINGS):
                recommendation = ImprovementRecommendation(
                    recommendation_id=f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_confidence",
                    strategy=ImprovementStrategy.RETRAIN_EMBEDDINGS,
                    priority='medium',
                    confidence=0.7,
                    description=f"Low average confidence score ({avg_confidence:.2f}) suggests need for embedding retraining",
                    expected_impact='Improved response accuracy and confidence',
                    implementation_effort='high',
                    supporting_data={'avg_confidence': avg_confidence, 'threshold': 0.7}
                )
                self.improvement_recommendations.append(recommendation)
            
            if avg_response_time > 3.0 and not self._has_recent_recommendation(ImprovementStrategy.OPTIMIZE_QUERY_PROCESSING):
                recommendation = ImprovementRecommendation(
                    recommendation_id=f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_performance",
                    strategy=ImprovementStrategy.OPTIMIZE_QUERY_PROCESSING,
                    priority='medium',
                    confidence=0.6,
                    description=f"High average response time ({avg_response_time:.2f}s) suggests need for optimization",
                    expected_impact='Faster response times and better user experience',
                    implementation_effort='medium',
                    supporting_data={'avg_response_time': avg_response_time, 'threshold': 3.0}
                )
                self.improvement_recommendations.append(recommendation)
            
        except Exception as e:
            self.logger.error(f"Error updating improvement recommendations: {e}")

    def _has_recent_recommendation(self, strategy: ImprovementStrategy, days: int = 7) -> bool:
        """Check if there's a recent recommendation for the given strategy."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for rec in self.improvement_recommendations:
            if rec.strategy == strategy and rec.created_at >= cutoff_date:
                return True
        
        return False

    def _monitor_ab_tests(self) -> None:
        """Monitor active A/B tests and check for completion conditions."""
        try:
            current_time = datetime.now()
            
            for test_id, test in list(self.active_ab_tests.items()):
                # Check if test should be completed
                if current_time >= test.end_date:
                    self.logger.info(f"A/B test {test_id} reached end date, analyzing results")
                    
                    # Analyze results and determine if treatment should be implemented
                    results = self.analyze_ab_test_results(test_id)
                    implement_treatment = results.get('recommendation') == 'implement_treatment'
                    
                    self.complete_ab_test(test_id, implement_treatment)
            
        except Exception as e:
            self.logger.error(f"Error monitoring A/B tests: {e}")

    def _cleanup_old_data(self) -> None:
        """Clean up old improvement data to prevent storage bloat."""
        try:
            # Clean up old recommendations (keep only last 30 days)
            cutoff_date = datetime.now() - timedelta(days=30)
            self.improvement_recommendations = [
                rec for rec in self.improvement_recommendations
                if rec.created_at >= cutoff_date
            ]
            
            # Clean up old improvement history (keep only last 90 days)
            history_cutoff = datetime.now() - timedelta(days=90)
            self.improvement_history = [
                record for record in self.improvement_history
                if datetime.fromisoformat(record['timestamp']) >= history_cutoff
            ]
            
            # Clean up old embedding versions (keep only last 10)
            if len(self.embedding_versions) > 10:
                # Sort by creation date and keep the 10 most recent
                sorted_versions = sorted(self.embedding_versions, key=lambda v: v.created_at, reverse=True)
                versions_to_remove = sorted_versions[10:]
                
                for version in versions_to_remove:
                    # Remove backup files
                    if version.backup_path and os.path.exists(version.backup_path):
                        try:
                            os.remove(version.backup_path)
                        except Exception as e:
                            self.logger.warning(f"Failed to remove backup {version.backup_path}: {e}")
                
                self.embedding_versions = sorted_versions[:10]
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")

    def get_improvement_recommendations(self, 
                                     strategy: Optional[ImprovementStrategy] = None,
                                     priority: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get improvement recommendations with optional filtering.
        
        Args:
            strategy: Optional strategy filter
            priority: Optional priority filter ('high', 'medium', 'low')
            
        Returns:
            List of improvement recommendations
        """
        try:
            recommendations = self.improvement_recommendations
            
            # Apply filters
            if strategy:
                recommendations = [rec for rec in recommendations if rec.strategy == strategy]
            
            if priority:
                recommendations = [rec for rec in recommendations if rec.priority == priority]
            
            # Convert to dictionaries and sort by priority and confidence
            priority_order = {'high': 3, 'medium': 2, 'low': 1}
            
            recommendations_data = []
            for rec in recommendations:
                rec_data = {
                    'recommendation_id': rec.recommendation_id,
                    'strategy': rec.strategy.value,
                    'priority': rec.priority,
                    'confidence': rec.confidence,
                    'description': rec.description,
                    'expected_impact': rec.expected_impact,
                    'implementation_effort': rec.implementation_effort,
                    'supporting_data': rec.supporting_data,
                    'created_at': rec.created_at.isoformat(),
                    'age_days': (datetime.now() - rec.created_at).days
                }
                recommendations_data.append(rec_data)
            
            # Sort by priority (high first) then by confidence (high first)
            recommendations_data.sort(
                key=lambda x: (priority_order.get(x['priority'], 0), x['confidence']),
                reverse=True
            )
            
            return recommendations_data
            
        except Exception as e:
            self.logger.error(f"Failed to get improvement recommendations: {e}")
            return []

    def get_ab_test_status(self) -> Dict[str, Any]:
        """
        Get status of all A/B tests.
        
        Returns:
            Dictionary containing A/B test status information
        """
        try:
            return {
                'active_tests': {
                    test_id: {
                        'name': test.name,
                        'strategy': test.strategy.value,
                        'start_date': test.start_date.isoformat(),
                        'end_date': test.end_date.isoformat(),
                        'traffic_split': test.traffic_split,
                        'days_remaining': (test.end_date - datetime.now()).days
                    }
                    for test_id, test in self.active_ab_tests.items()
                },
                'completed_tests': len(self.ab_test_history),
                'total_user_assignments': len(self.ab_test_assignments)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get A/B test status: {e}")
            return {'error': str(e)}

    def get_system_improvement_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive system improvement report.
        
        Returns:
            Dictionary containing system improvement status and recommendations
        """
        try:
            current_version = self._get_current_version()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'embedding_versions': {
                    'current_version': self.current_embedding_version,
                    'total_versions': len(self.embedding_versions),
                    'current_version_info': {
                        'created_at': current_version.created_at.isoformat() if current_version else None,
                        'model_name': current_version.model_name if current_version else None,
                        'total_faqs': current_version.total_faqs if current_version else 0,
                        'performance_metrics': current_version.performance_metrics if current_version else {}
                    } if current_version else None
                },
                'ab_testing': self.get_ab_test_status(),
                'improvement_recommendations': {
                    'total_recommendations': len(self.improvement_recommendations),
                    'high_priority': len([r for r in self.improvement_recommendations if r.priority == 'high']),
                    'recent_recommendations': len([
                        r for r in self.improvement_recommendations 
                        if (datetime.now() - r.created_at).days <= 7
                    ])
                },
                'retraining_config': self.retraining_config,
                'improvement_history': {
                    'total_improvements': len(self.improvement_history),
                    'recent_improvements': len([
                        record for record in self.improvement_history
                        if (datetime.now() - datetime.fromisoformat(record['timestamp'])).days <= 30
                    ])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate system improvement report: {e}")
            return {'error': str(e)}

    def update_retraining_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Update retraining configuration.
        
        Args:
            new_config: New configuration parameters
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            self.retraining_config.update(new_config)
            self._save_persistent_data()
            
            self.logger.info(f"Updated retraining configuration: {new_config}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update retraining config: {e}")
            return False

    def stop_monitoring(self) -> None:
        """Stop the system improvement monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        # Save final data
        self._save_persistent_data()
        
        self.logger.info("System improvement monitoring stopped")