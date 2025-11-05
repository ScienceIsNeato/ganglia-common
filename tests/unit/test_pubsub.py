"""Unit tests for the pubsub system.

This module contains tests for the publish-subscribe system used for
asynchronous communication between components.
"""

import unittest
import time
from unittest.mock import MagicMock
from ganglia_common.pubsub import get_pubsub, Event, EventType, PubSub


class TestPubSub(unittest.TestCase):
    """Test cases for the pubsub system."""

    def setUp(self):
        """Set up test fixtures."""
        # Get a fresh pubsub instance for each test
        self.pubsub = get_pubsub()
        self.pubsub.start()

    def tearDown(self):
        """Tear down test fixtures."""
        self.pubsub.stop()

    def test_subscribe_and_publish(self):
        """Test subscribing to events and publishing events."""
        # Create a mock callback
        callback = MagicMock()

        # Subscribe to an event type
        self.pubsub.subscribe(EventType.STORY_INFO_NEEDED, callback)

        # Create and publish an event
        event_data = {"info_type": "story_idea", "prompt": "Test prompt"}
        event = Event(
            event_type=EventType.STORY_INFO_NEEDED,
            data=event_data,
            source="test",
            target="user123"
        )
        self.pubsub.publish(event)

        # Wait for the event to be processed
        time.sleep(0.1)

        # Verify the callback was called with the correct event
        callback.assert_called_once()
        received_event = callback.call_args[0][0]
        self.assertEqual(received_event.event_type, EventType.STORY_INFO_NEEDED)
        self.assertEqual(received_event.data, event_data)
        self.assertEqual(received_event.source, "test")
        self.assertEqual(received_event.target, "user123")

    def test_unsubscribe(self):
        """Test unsubscribing from events."""
        # Create a mock callback
        callback = MagicMock()

        # Subscribe to an event type
        self.pubsub.subscribe(EventType.STORY_INFO_NEEDED, callback)

        # Unsubscribe from the event type
        self.pubsub.unsubscribe(EventType.STORY_INFO_NEEDED, callback)

        # Create and publish an event
        event = Event(
            event_type=EventType.STORY_INFO_NEEDED,
            data={"info_type": "story_idea"},
            source="test"
        )
        self.pubsub.publish(event)

        # Wait for the event to be processed
        time.sleep(0.1)

        # Verify the callback was not called
        callback.assert_not_called()

    def test_multiple_subscribers(self):
        """Test multiple subscribers for the same event type."""
        # Create mock callbacks
        callback1 = MagicMock()
        callback2 = MagicMock()

        # Subscribe both callbacks to the same event type
        self.pubsub.subscribe(EventType.STORY_INFO_NEEDED, callback1)
        self.pubsub.subscribe(EventType.STORY_INFO_NEEDED, callback2)

        # Create and publish an event
        event = Event(
            event_type=EventType.STORY_INFO_NEEDED,
            data={"info_type": "story_idea"},
            source="test"
        )
        self.pubsub.publish(event)

        # Wait for the event to be processed
        time.sleep(0.1)

        # Verify both callbacks were called
        callback1.assert_called_once()
        callback2.assert_called_once()

    def test_targeted_events(self):
        """Test that targeted events only reach the intended target."""
        # Create mock callbacks
        callback1 = MagicMock()
        callback2 = MagicMock()

        # Subscribe both callbacks to the same event type
        self.pubsub.subscribe(EventType.STORY_INFO_NEEDED, callback1)
        self.pubsub.subscribe(EventType.STORY_INFO_NEEDED, callback2)

        # Create a targeted event
        event = Event(
            event_type=EventType.STORY_INFO_NEEDED,
            data={"info_type": "story_idea"},
            source="test",
            target="user123"
        )

        # Set up the callbacks to check the target
        def check_target1(event):
            if event.target == "user123":
                callback1.real_call(event)

        def check_target2(event):
            if event.target == "user456":
                callback2.real_call(event)

        callback1.side_effect = check_target1
        callback1.real_call = MagicMock()
        callback2.side_effect = check_target2
        callback2.real_call = MagicMock()

        # Publish the event
        self.pubsub.publish(event)

        # Wait for the event to be processed
        time.sleep(0.1)

        # Verify only the callback with the matching target was called
        callback1.real_call.assert_called_once()
        callback2.real_call.assert_not_called()

    def test_event_types(self):
        """Test that events are only delivered to subscribers of the correct type."""
        # Create mock callbacks
        callback1 = MagicMock()
        callback2 = MagicMock()

        # Subscribe to different event types
        self.pubsub.subscribe(EventType.STORY_INFO_NEEDED, callback1)
        self.pubsub.subscribe(EventType.TTV_PROCESS_STARTED, callback2)

        # Create and publish an event of one type
        event = Event(
            event_type=EventType.STORY_INFO_NEEDED,
            data={"info_type": "story_idea"},
            source="test"
        )
        self.pubsub.publish(event)

        # Wait for the event to be processed
        time.sleep(0.1)

        # Verify only the callback for the correct event type was called
        callback1.assert_called_once()
        callback2.assert_not_called()


if __name__ == "__main__":
    unittest.main()
