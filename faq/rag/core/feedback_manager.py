from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import json
import os
from ..interfaces.base import FeedbackManagerInterface
from ..utils.logging import get_rag_logger


class FeedbackManager(FeedbackManagerInterface):
    """
    Comprehensive user feedback management implementation.
    
    Provides detailed feedback tracking, analysis, and insights for continuous
    system improvement with persistent storage and advanced analytics.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.logger = get_rag_logger('feedback_manager')
        
        # Core feedback storage
        self.feedback_entries: List[Dict[str, Any]] = []
        
        # Feedback analytics
        self.feedback_analytics = {
            'rating_trends': [],
            'user_satisfaction_scores': defaultdict(list),
            'feedback_categories': Counter(),
            'improvement_suggestions': [],
            'response_quality_feedback': []
        }
        
        # User behavior tracking
        self.user_patterns = {
            'frequent_users': Counter(),
            'user_satisfaction_history': defaultdict(list),
            'feedback_frequency': defaultdict(int)
        }
        
        # Persistent storage setup
        self.storage_path = storage_path or "feedback_data"
        self._ensure_storage_directory()
        self._load_persistent_feedback()
        
        self.logger.info("Enhanced FeedbackManager initialized with comprehensive tracking.")

    def _ensure_storage_directory(self) -> None:
        """Ensure feedback storage directory exists."""
        try:
            os.makedirs(self.storage_path, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Failed to create feedback storage directory: {e}")
    
    def _load_persistent_feedback(self) -> None:
        """Load persistent feedback data from storage."""
        try:
            feedback_file = os.path.join(self.storage_path, "feedback_entries.json")
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r') as f:
                    self.feedback_entries = json.load(f)
            
            analytics_file = os.path.join(self.storage_path, "feedback_analytics.json")
            if os.path.exists(analytics_file):
                with open(analytics_file, 'r') as f:
                    data = json.load(f)
                    self.feedback_analytics['feedback_categories'] = Counter(data.get('feedback_categories', {}))
                    self.user_patterns['frequent_users'] = Counter(data.get('frequent_users', {}))
            
            self.logger.info(f"Loaded {len(self.feedback_entries)} feedback entries from storage")
        except Exception as e:
            self.logger.warning(f"Failed to load persistent feedback data: {e}")
    
    def _save_persistent_feedback(self) -> None:
        """Save feedback data to persistent storage."""
        try:
            # Save feedback entries (keep only recent ones to prevent file bloat)
            feedback_file = os.path.join(self.storage_path, "feedback_entries.json")
            recent_feedback = self.feedback_entries[-1000:] if len(self.feedback_entries) > 1000 else self.feedback_entries
            with open(feedback_file, 'w') as f:
                json.dump(recent_feedback, f, indent=2)
            
            # Save analytics data
            analytics_file = os.path.join(self.storage_path, "feedback_analytics.json")
            analytics_data = {
                'feedback_categories': dict(self.feedback_analytics['feedback_categories']),
                'frequent_users': dict(self.user_patterns['frequent_users'])
            }
            with open(analytics_file, 'w') as f:
                json.dump(analytics_data, f, indent=2)
            
            self.logger.debug("Saved persistent feedback data")
        except Exception as e:
            self.logger.warning(f"Failed to save persistent feedback data: {e}")

    def submit_feedback(self, query_id: str, user_id: str, rating: int, comments: Optional[str] = None) -> None:
        """
        Submit comprehensive user feedback with detailed tracking and analysis.
        
        Args:
            query_id: Unique identifier for the query being rated
            user_id: Identifier for the user providing feedback
            rating: Numerical rating (typically 1-5 scale)
            comments: Optional textual feedback comments
        """
        timestamp = datetime.now()
        
        # Validate rating
        if not (1 <= rating <= 5):
            self.logger.warning(f"Invalid rating {rating} submitted by user {user_id}. Using clamped value.")
            rating = max(1, min(5, rating))
        
        # Create comprehensive feedback entry
        feedback_entry = {
            "feedback_id": f"fb_{timestamp.timestamp()}_{user_id}",
            "query_id": query_id,
            "user_id": user_id,
            "rating": rating,
            "comments": comments,
            "timestamp": timestamp.isoformat(),
            "feedback_metadata": {
                "comment_length": len(comments) if comments else 0,
                "has_detailed_feedback": bool(comments and len(comments) > 20),
                "sentiment": self._analyze_comment_sentiment(comments) if comments else "neutral",
                "feedback_category": self._categorize_feedback(rating, comments)
            }
        }
        
        # Store feedback
        self.feedback_entries.append(feedback_entry)
        
        # Update analytics
        self._update_feedback_analytics(feedback_entry)
        
        # Update user patterns
        self._update_user_patterns(user_id, rating, timestamp)
        
        # Periodic data persistence (every 5 feedback entries)
        if len(self.feedback_entries) % 5 == 0:
            self._save_persistent_feedback()
        
        self.logger.info(f"Comprehensive feedback submitted for query {query_id} by user {user_id} "
                        f"with rating {rating} (category: {feedback_entry['feedback_metadata']['feedback_category']})")
    
    def _analyze_comment_sentiment(self, comments: str) -> str:
        """Simple sentiment analysis of feedback comments."""
        if not comments:
            return "neutral"
        
        comments_lower = comments.lower()
        
        # Simple keyword-based sentiment analysis
        positive_words = ["good", "great", "excellent", "helpful", "accurate", "fast", "perfect", "love", "amazing"]
        negative_words = ["bad", "terrible", "wrong", "slow", "useless", "hate", "awful", "poor", "disappointing"]
        
        positive_count = sum(1 for word in positive_words if word in comments_lower)
        negative_count = sum(1 for word in negative_words if word in comments_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _categorize_feedback(self, rating: int, comments: Optional[str]) -> str:
        """Categorize feedback based on rating and comments."""
        if rating >= 4:
            if comments and any(word in comments.lower() for word in ["fast", "quick", "speed"]):
                return "performance_praise"
            elif comments and any(word in comments.lower() for word in ["accurate", "correct", "right"]):
                return "accuracy_praise"
            else:
                return "general_satisfaction"
        elif rating <= 2:
            if comments and any(word in comments.lower() for word in ["slow", "timeout", "wait"]):
                return "performance_complaint"
            elif comments and any(word in comments.lower() for word in ["wrong", "incorrect", "bad"]):
                return "accuracy_complaint"
            elif comments and any(word in comments.lower() for word in ["confusing", "unclear", "hard"]):
                return "usability_complaint"
            else:
                return "general_dissatisfaction"
        else:
            return "neutral_feedback"
    
    def _update_feedback_analytics(self, feedback_entry: Dict[str, Any]) -> None:
        """Update feedback analytics with new entry."""
        # Update rating trends
        self.feedback_analytics['rating_trends'].append({
            'rating': feedback_entry['rating'],
            'timestamp': feedback_entry['timestamp']
        })
        
        # Update feedback categories
        category = feedback_entry['feedback_metadata']['feedback_category']
        self.feedback_analytics['feedback_categories'][category] += 1
        
        # Track improvement suggestions
        if feedback_entry['comments'] and feedback_entry['rating'] <= 3:
            self.feedback_analytics['improvement_suggestions'].append({
                'suggestion': feedback_entry['comments'],
                'rating': feedback_entry['rating'],
                'timestamp': feedback_entry['timestamp'],
                'category': category
            })
        
        # Track response quality feedback
        self.feedback_analytics['response_quality_feedback'].append({
            'query_id': feedback_entry['query_id'],
            'rating': feedback_entry['rating'],
            'sentiment': feedback_entry['feedback_metadata']['sentiment'],
            'timestamp': feedback_entry['timestamp']
        })
    
    def _update_user_patterns(self, user_id: str, rating: int, timestamp: datetime) -> None:
        """Update user behavior patterns."""
        # Track frequent users
        self.user_patterns['frequent_users'][user_id] += 1
        
        # Track user satisfaction history
        self.user_patterns['user_satisfaction_history'][user_id].append({
            'rating': rating,
            'timestamp': timestamp.isoformat()
        })
        
        # Track feedback frequency
        date_key = timestamp.date().isoformat()
        self.user_patterns['feedback_frequency'][date_key] += 1

    def get_feedback(self, query_id: Optional[str] = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve feedback entries with optional filtering.
        
        Args:
            query_id: Optional query ID to filter by
            user_id: Optional user ID to filter by
            
        Returns:
            List of matching feedback entries
        """
        filtered_feedback = self.feedback_entries
        
        if query_id:
            filtered_feedback = [f for f in filtered_feedback if f["query_id"] == query_id]
        
        if user_id:
            filtered_feedback = [f for f in filtered_feedback if f["user_id"] == user_id]
        
        # Sort by timestamp (most recent first)
        filtered_feedback.sort(key=lambda x: x["timestamp"], reverse=True)
        
        self.logger.debug(f"Retrieved {len(filtered_feedback)} feedback entries "
                         f"(query_id: {query_id}, user_id: {user_id})")
        
        return filtered_feedback

    def analyze_feedback(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Perform comprehensive feedback analysis with detailed insights.
        
        Args:
            start_date: Optional start date for analysis period
            end_date: Optional end date for analysis period
            
        Returns:
            Dictionary containing detailed feedback analysis
        """
        self.logger.info("Performing comprehensive feedback analysis...")
        
        # Filter feedback by date range
        filtered_feedback = []
        for feedback in self.feedback_entries:
            feedback_time = datetime.fromisoformat(feedback["timestamp"])
            if (start_date is None or feedback_time >= start_date) and \
               (end_date is None or feedback_time <= end_date):
                filtered_feedback.append(feedback)
        
        if not filtered_feedback:
            return {
                "analysis_period": {
                    "start": start_date.isoformat() if start_date else "beginning",
                    "end": end_date.isoformat() if end_date else "now"
                },
                "total_feedback_entries": 0,
                "message": "No feedback entries found for the specified period"
            }
        
        # Basic statistics
        total_ratings = len(filtered_feedback)
        ratings = [f["rating"] for f in filtered_feedback]
        sum_ratings = sum(ratings)
        average_rating = sum_ratings / total_ratings
        
        # Rating distribution
        rating_distribution = Counter(ratings)
        positive_feedback = len([r for r in ratings if r >= 4])
        negative_feedback = len([r for r in ratings if r <= 2])
        neutral_feedback = total_ratings - positive_feedback - negative_feedback
        
        # Sentiment analysis
        sentiments = [f["feedback_metadata"]["sentiment"] for f in filtered_feedback]
        sentiment_distribution = Counter(sentiments)
        
        # Category analysis
        categories = [f["feedback_metadata"]["feedback_category"] for f in filtered_feedback]
        category_distribution = Counter(categories)
        
        # User engagement analysis
        unique_users = len(set([f["user_id"] for f in filtered_feedback]))
        feedback_per_user = total_ratings / unique_users if unique_users > 0 else 0
        
        # Temporal analysis
        temporal_analysis = self._analyze_feedback_trends(filtered_feedback)
        
        # Comment analysis
        comments_analysis = self._analyze_feedback_comments(filtered_feedback)
        
        # Improvement opportunities
        improvement_opportunities = self._identify_improvement_opportunities(filtered_feedback)
        
        # User satisfaction trends
        satisfaction_trends = self._analyze_user_satisfaction_trends(filtered_feedback)
        
        return {
            "analysis_period": {
                "start": start_date.isoformat() if start_date else "beginning",
                "end": end_date.isoformat() if end_date else "now",
                "duration_days": (end_date - start_date).days if start_date and end_date else None
            },
            "overall_statistics": {
                "total_feedback_entries": total_ratings,
                "average_rating": round(average_rating, 2),
                "median_rating": sorted(ratings)[len(ratings)//2] if ratings else 0,
                "rating_std_dev": self._calculate_std_dev(ratings),
                "unique_users": unique_users,
                "feedback_per_user": round(feedback_per_user, 2)
            },
            "rating_distribution": {
                "detailed": dict(rating_distribution),
                "summary": {
                    "positive_feedback_count": positive_feedback,
                    "negative_feedback_count": negative_feedback,
                    "neutral_feedback_count": neutral_feedback,
                    "positive_percentage": round((positive_feedback / total_ratings) * 100, 1),
                    "negative_percentage": round((negative_feedback / total_ratings) * 100, 1)
                }
            },
            "sentiment_analysis": {
                "distribution": dict(sentiment_distribution),
                "positive_sentiment_rate": sentiment_distribution.get("positive", 0) / total_ratings,
                "negative_sentiment_rate": sentiment_distribution.get("negative", 0) / total_ratings
            },
            "feedback_categories": {
                "distribution": dict(category_distribution.most_common()),
                "top_complaint_category": category_distribution.most_common(1)[0] if category_distribution else None,
                "top_praise_category": self._get_top_praise_category(category_distribution)
            },
            "temporal_analysis": temporal_analysis,
            "comments_analysis": comments_analysis,
            "improvement_opportunities": improvement_opportunities,
            "user_satisfaction_trends": satisfaction_trends,
            "data_quality": {
                "feedback_with_comments": len([f for f in filtered_feedback if f["comments"]]),
                "detailed_feedback_rate": len([f for f in filtered_feedback if f["feedback_metadata"]["has_detailed_feedback"]]) / total_ratings,
                "average_comment_length": sum([f["feedback_metadata"]["comment_length"] for f in filtered_feedback]) / total_ratings
            }
        }
    
    def _calculate_std_dev(self, ratings: List[int]) -> float:
        """Calculate standard deviation of ratings."""
        if len(ratings) < 2:
            return 0.0
        
        mean = sum(ratings) / len(ratings)
        variance = sum((r - mean) ** 2 for r in ratings) / len(ratings)
        return variance ** 0.5
    
    def _get_top_praise_category(self, category_distribution: Counter) -> Optional[str]:
        """Get the most common praise category."""
        praise_categories = [cat for cat in category_distribution.keys() if "praise" in cat]
        if praise_categories:
            return max(praise_categories, key=lambda x: category_distribution[x])
        return None
    
    def _analyze_feedback_trends(self, filtered_feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze feedback trends over time."""
        if len(filtered_feedback) < 5:
            return {"trend": "insufficient_data"}
        
        # Sort by timestamp
        sorted_feedback = sorted(filtered_feedback, key=lambda x: x["timestamp"])
        
        # Split into periods and compare
        mid_point = len(sorted_feedback) // 2
        early_period = sorted_feedback[:mid_point]
        recent_period = sorted_feedback[mid_point:]
        
        early_avg = sum([f["rating"] for f in early_period]) / len(early_period)
        recent_avg = sum([f["rating"] for f in recent_period]) / len(recent_period)
        
        trend = "improving" if recent_avg > early_avg + 0.2 else \
                "declining" if recent_avg < early_avg - 0.2 else "stable"
        
        return {
            "trend": trend,
            "early_period_avg": round(early_avg, 2),
            "recent_period_avg": round(recent_avg, 2),
            "trend_magnitude": round(abs(recent_avg - early_avg), 2)
        }
    
    def _analyze_feedback_comments(self, filtered_feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze textual feedback comments."""
        comments = [f["comments"] for f in filtered_feedback if f["comments"]]
        
        if not comments:
            return {"total_comments": 0, "analysis": "no_comments_available"}
        
        # Extract common words from negative feedback
        negative_comments = [
            f["comments"] for f in filtered_feedback 
            if f["comments"] and f["rating"] <= 2
        ]
        
        positive_comments = [
            f["comments"] for f in filtered_feedback 
            if f["comments"] and f["rating"] >= 4
        ]
        
        return {
            "total_comments": len(comments),
            "negative_comments": len(negative_comments),
            "positive_comments": len(positive_comments),
            "average_comment_length": sum(len(c) for c in comments) / len(comments),
            "detailed_feedback_count": len([c for c in comments if len(c) > 50]),
            "common_negative_themes": self._extract_common_themes(negative_comments),
            "common_positive_themes": self._extract_common_themes(positive_comments)
        }
    
    def _extract_common_themes(self, comments: List[str]) -> List[str]:
        """Extract common themes from comments (simple keyword extraction)."""
        if not comments:
            return []
        
        # Simple word frequency analysis
        all_words = []
        for comment in comments:
            words = comment.lower().split()
            # Filter out common words and short words
            filtered_words = [w for w in words if len(w) > 3 and w not in 
                            ["this", "that", "with", "from", "they", "were", "been", "have", "will"]]
            all_words.extend(filtered_words)
        
        word_counts = Counter(all_words)
        return [word for word, count in word_counts.most_common(5) if count > 1]
    
    def _identify_improvement_opportunities(self, filtered_feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify specific improvement opportunities from feedback."""
        opportunities = []
        
        # Analyze complaint categories
        complaint_categories = Counter([
            f["feedback_metadata"]["feedback_category"] 
            for f in filtered_feedback 
            if "complaint" in f["feedback_metadata"]["feedback_category"]
        ])
        
        for category, count in complaint_categories.most_common(3):
            opportunities.append({
                "area": category.replace("_complaint", ""),
                "frequency": count,
                "priority": "high" if count > len(filtered_feedback) * 0.1 else "medium",
                "description": f"Multiple users reported issues with {category.replace('_complaint', '').replace('_', ' ')}"
            })
        
        # Analyze low-rating feedback for patterns
        low_rating_feedback = [f for f in filtered_feedback if f["rating"] <= 2]
        if len(low_rating_feedback) > len(filtered_feedback) * 0.2:  # More than 20% negative
            opportunities.append({
                "area": "overall_satisfaction",
                "frequency": len(low_rating_feedback),
                "priority": "high",
                "description": "High percentage of negative feedback indicates systemic issues"
            })
        
        return opportunities
    
    def _analyze_user_satisfaction_trends(self, filtered_feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user satisfaction trends and patterns."""
        # Group feedback by user
        user_feedback = defaultdict(list)
        for feedback in filtered_feedback:
            user_feedback[feedback["user_id"]].append(feedback)
        
        # Analyze user satisfaction patterns
        improving_users = 0
        declining_users = 0
        consistent_users = 0
        
        for user_id, user_feedbacks in user_feedback.items():
            if len(user_feedbacks) < 2:
                continue
            
            # Sort by timestamp
            user_feedbacks.sort(key=lambda x: x["timestamp"])
            first_rating = user_feedbacks[0]["rating"]
            last_rating = user_feedbacks[-1]["rating"]
            
            if last_rating > first_rating + 0.5:
                improving_users += 1
            elif last_rating < first_rating - 0.5:
                declining_users += 1
            else:
                consistent_users += 1
        
        return {
            "users_with_multiple_feedback": len([u for u in user_feedback.values() if len(u) > 1]),
            "improving_users": improving_users,
            "declining_users": declining_users,
            "consistent_users": consistent_users,
            "most_active_users": [
                {"user_id": user_id, "feedback_count": len(feedbacks)}
                for user_id, feedbacks in sorted(user_feedback.items(), 
                                               key=lambda x: len(x[1]), reverse=True)[:5]
            ]
        }
    
    def get_user_feedback_summary(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive feedback summary for a specific user."""
        user_feedback = [f for f in self.feedback_entries if f["user_id"] == user_id]
        
        if not user_feedback:
            return {"user_id": user_id, "total_feedback": 0, "message": "No feedback found for user"}
        
        ratings = [f["rating"] for f in user_feedback]
        
        return {
            "user_id": user_id,
            "total_feedback": len(user_feedback),
            "average_rating": sum(ratings) / len(ratings),
            "rating_range": {"min": min(ratings), "max": max(ratings)},
            "feedback_frequency": len(user_feedback) / max(1, (datetime.now() - datetime.fromisoformat(user_feedback[0]["timestamp"])).days),
            "recent_feedback": sorted(user_feedback, key=lambda x: x["timestamp"], reverse=True)[:3],
            "satisfaction_trend": self._calculate_user_satisfaction_trend(user_feedback)
        }
    
    def _calculate_user_satisfaction_trend(self, user_feedback: List[Dict[str, Any]]) -> str:
        """Calculate satisfaction trend for a specific user."""
        if len(user_feedback) < 2:
            return "insufficient_data"
        
        sorted_feedback = sorted(user_feedback, key=lambda x: x["timestamp"])
        recent_ratings = [f["rating"] for f in sorted_feedback[-3:]]  # Last 3 ratings
        early_ratings = [f["rating"] for f in sorted_feedback[:3]]    # First 3 ratings
        
        recent_avg = sum(recent_ratings) / len(recent_ratings)
        early_avg = sum(early_ratings) / len(early_ratings)
        
        if recent_avg > early_avg + 0.5:
            return "improving"
        elif recent_avg < early_avg - 0.5:
            return "declining"
        else:
            return "stable"
    
    def export_feedback_data(self, export_path: str, include_comments: bool = True) -> bool:
        """
        Export feedback data to JSON file.
        
        Args:
            export_path: Path to export file
            include_comments: Whether to include user comments in export
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            export_data = {
                "feedback_entries": self.feedback_entries if include_comments else [
                    {k: v for k, v in entry.items() if k != "comments"} 
                    for entry in self.feedback_entries
                ],
                "analytics_summary": {
                    "total_entries": len(self.feedback_entries),
                    "feedback_categories": dict(self.feedback_analytics['feedback_categories']),
                    "frequent_users": dict(self.user_patterns['frequent_users'].most_common(10))
                },
                "export_metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "include_comments": include_comments,
                    "total_users": len(set([f["user_id"] for f in self.feedback_entries]))
                }
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Feedback data exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export feedback data: {e}")
            return False