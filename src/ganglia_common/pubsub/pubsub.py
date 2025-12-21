"""PubSub system for asynchronous inter-module communication.

This module provides a simple publish-subscribe pattern implementation
for decoupled communication between different components of the GANGLIA system,
particularly enabling the integration of chatbot and TTV functionalities.
"""

from typing import Dict, List, Callable, Any
from enum import Enum, auto
import uuid
import queue
import threading
import time
from ganglia_common.logger import Logger


class EventType(Enum):
    """Types of events that can be published on the pubsub system."""

    # Story generation related events
    STORY_INFO_NEEDED = auto()  # Need to gather specific story info
    STORY_INFO_RECEIVED = auto()  # User provided requested story info
    STORY_CONFIG_COMPLETE = auto()  # All info gathered, config ready
    TTV_CONFIG_GENERATED = auto()  # TTV configuration file has been generated

    # Video generation related events
    TTV_PROCESS_STARTED = auto()  # TTV process has started
    TTV_PROCESS_PROGRESS = auto()  # TTV process progress update
    TTV_PROCESS_COMPLETED = auto()  # TTV process has completed
    TTV_PROCESS_FAILED = auto()  # TTV process failed

    # Conversation related events
    USER_PROFILE_UPDATED = auto()  # User profile information was updated
    CONVERSATION_STARTED = auto()  # New conversation started
    CONVERSATION_ENDED = auto()  # Conversation ended

    # General purpose events
    CUSTOM = auto()  # Custom event type for extensibility


class Event:
    """Represents an event in the pubsub system."""

    def __init__(
        self,
        event_type: EventType,
        data: Dict[str, Any] = None,
        source: str = None,
        target: str = None,
    ):
        """
        Initialize a new event.

        Args:
            event_type: Type of the event from EventType enum
            data: Optional dictionary of data associated with the event
            source: Optional identifier of the component that published the event
            target: Optional identifier of the intended target component (None = broadcast)
        """
        self.id = str(uuid.uuid4())
        self.event_type = event_type
        self.data = data or {}
        self.source = source
        self.target = target
        self.timestamp = time.time()  # Event creation time

    def __str__(self):
        return (
            f"Event({self.event_type.name}, source={self.source}, target={self.target})"
        )


class PubSub:
    """A simple publish-subscribe system for asynchronous communication."""

    def __init__(self):
        """Initialize the pubsub system."""
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {}
        self._event_queue = queue.Queue()
        self._running = False
        self._thread = None

        # Initialize the event types
        for event_type in EventType:
            self._subscribers[event_type] = []

    def start(self):
        """Start the pubsub event processing thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._process_events, daemon=True)
        self._thread.start()
        Logger.print_debug("PubSub system started")

    def stop(self):
        """Stop the pubsub event processing thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        Logger.print_debug("PubSub system stopped")

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """
        Subscribe to a specific event type.

        Args:
            event_type: The event type to subscribe to
            callback: Function to call when event occurs, takes Event as parameter
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        self._subscribers[event_type].append(callback)
        Logger.print_debug(f"Subscribed to {event_type.name} events")

    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """
        Unsubscribe from a specific event type.

        Args:
            event_type: The event type to unsubscribe from
            callback: The callback function to remove
        """
        if (
            event_type in self._subscribers
            and callback in self._subscribers[event_type]
        ):
            self._subscribers[event_type].remove(callback)
            Logger.print_debug(f"Unsubscribed from {event_type.name} events")

    def publish(self, event: Event):
        """
        Publish an event to all subscribers.

        Args:
            event: The event to publish
        """
        self._event_queue.put(event)
        Logger.print_debug(f"Published event: {event}")

    def _process_events(self):
        """Process events from the queue and dispatch to subscribers."""
        while self._running:
            try:
                event = self._event_queue.get(timeout=0.1)
                self._dispatch_event(event)
                self._event_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                Logger.print_error(f"Error processing event: {e}")

    def _dispatch_event(self, event: Event):
        """
        Dispatch an event to all subscribers of its type.

        Args:
            event: The event to dispatch
        """
        if event.event_type in self._subscribers:
            for callback in self._subscribers[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    Logger.print_error(f"Error in event callback: {e}")


# Singleton instance
_instance = None


def get_pubsub():
    """Get the singleton PubSub instance."""
    global _instance
    if _instance is None:
        _instance = PubSub()
        _instance.start()
    return _instance
