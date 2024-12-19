from threading import Event, Thread
from typing import Generator

import pytest

from hsm.core.data_management import DataLockError, DataManager


@pytest.fixture
def data_manager() -> DataManager:
    return DataManager()


@pytest.fixture
def populated_manager(data_manager: DataManager) -> DataManager:
    """Returns a DataManager with some preset test data."""
    with data_manager.access_data() as data:
        data.update({"key": "value", "counter": 0, "nested": {"inner": "data"}})
    return data_manager


@pytest.fixture
def sample_data() -> dict:
    """Returns a dictionary with various data types for testing."""
    return {
        "int": 42,
        "float": 3.14,
        "bool": True,
        "none": None,
        "list": [1, 2, 3],
        "dict": {"a": 1},
        "tuple": (1, 2),
        "set": {1, 2, 3},
    }


class TestDataManagerBasics:
    def test_init(self, data_manager: DataManager) -> None:
        """Test DataManager initialization."""
        assert isinstance(data_manager._data, dict)
        assert len(data_manager._data) == 0

    def test_basic_operations(self, data_manager: DataManager) -> None:
        """Test basic data operations within access_data context."""
        with data_manager.access_data() as data:
            data["key"] = "value"
            data["number"] = 42

        with data_manager.access_data() as data:
            assert data.get("key") == "value"
            assert data.get("number") == 42
            assert len(data) == 2


class TestDataManagerConcurrency:
    def test_concurrent_access(self, data_manager: DataManager) -> None:
        """Test concurrent access to DataManager."""
        thread_count = 10
        iterations = 100

        def increment_counter() -> None:
            for _ in range(iterations):
                with data_manager.access_data() as data:
                    current = data.get("counter", 0)
                    data["counter"] = current + 1

        threads = [Thread(target=increment_counter) for _ in range(thread_count)]

        # Start and join all threads
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        with data_manager.access_data() as data:
            assert data.get("counter") == thread_count * iterations

    def test_thread_blocking(self, data_manager: DataManager) -> None:
        """Test that threads properly block when lock is held."""
        lock_acquired = Event()
        lock_released = Event()

        def hold_lock() -> None:
            with data_manager.access_data() as data:
                lock_acquired.set()
                lock_released.wait(timeout=0.2)

        thread = Thread(target=hold_lock)
        thread.start()

        # Wait until first thread has the lock
        lock_acquired.wait()

        # Try to acquire lock - should timeout
        with pytest.raises(DataLockError) as exc_info:
            with data_manager.access_data(timeout=0.05):
                pass

        assert "Failed to acquire data lock" in str(exc_info.value)

        # Allow first thread to complete
        lock_released.set()
        thread.join()


class TestDataManagerExceptions:
    def test_exception_handling(self, populated_manager: DataManager) -> None:
        """Test exception handling within access_data context."""
        try:
            with populated_manager.access_data() as data:
                data["key"] = "modified"
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Verify data is rolled back
        with populated_manager.access_data() as data:
            assert data.get("key") == "value"


class TestDataManagerPerformance:
    def test_large_data_structure(self, data_manager: DataManager) -> None:
        """Test handling of large data structures."""
        large_dict = {str(i): "x" * 100 for i in range(1000)}

        with data_manager.access_data() as data:
            data["large_dict"] = large_dict

        # Verify we can access and modify
        with data_manager.access_data() as data:
            stored_dict = data.get("large_dict")
            assert len(stored_dict) == 1000
            assert len(stored_dict["0"]) == 100


def test_data_manager_init():
    """Test DataManager initialization."""
    dm = DataManager()
    assert isinstance(dm._data, dict)
    assert dm._data == {}

    dm = DataManager(lock_timeout=1.0)
    assert dm._lock_timeout == 1.0

    with pytest.raises(ValueError):
        DataManager(lock_timeout=-1.0)


def test_data_access_basic(data_manager):
    """Test basic data access operations."""
    with data_manager.access_data() as data:
        data["test_key"] = "test_value"

    with data_manager.access_data() as data:
        assert data["test_key"] == "test_value"


def test_data_isolation(populated_manager):
    """Test that data modifications are isolated within context."""
    with pytest.raises(RuntimeError):
        with populated_manager.access_data() as data:
            data["counter"] = 1
            raise RuntimeError("Test error")

    with populated_manager.access_data() as data:
        assert data["counter"] == 0


def test_snapshot(populated_manager):
    """Test snapshot functionality."""
    snapshot = populated_manager.get_data_snapshot()

    # Modify snapshot
    snapshot["key"] = "modified"
    snapshot["nested"]["inner"] = "modified"

    # Verify original data unchanged
    with populated_manager.access_data() as data:
        assert data["key"] == "value"
        assert data["nested"]["inner"] == "data"


def test_clear_data(populated_manager):
    """Test data clearing functionality."""
    populated_manager.clear_data()

    with populated_manager.access_data() as data:
        assert len(data) == 0


def test_timeout_behavior():
    """Test timeout behavior with infinite timeout."""
    dm = DataManager()

    with dm.access_data(timeout=float("inf")) as data:
        data["key"] = "value"

    assert dm.get_data_snapshot()["key"] == "value"


def test_concurrent_access_timeout():
    """Test that concurrent access properly times out."""
    dm = DataManager(lock_timeout=0.1)

    with dm.access_data():
        with pytest.raises(DataLockError):
            with dm.access_data():
                pass


def test_custom_timeout_override():
    """Test that access_data timeout parameter overrides default."""
    dm = DataManager(lock_timeout=0.1)

    with dm.access_data(timeout=0.2) as data:
        data["key"] = "value"

    assert dm.get_data_snapshot()["key"] == "value"


def test_nested_data_deepcopy():
    """Test that nested data structures are properly deep copied."""
    dm = DataManager()

    original_list = [1, 2, 3]
    original_dict = {"a": [4, 5, 6]}

    with dm.access_data() as data:
        data["list"] = original_list
        data["dict"] = original_dict

    # Modify original data
    original_list.append(4)
    original_dict["a"].append(7)

    with dm.access_data() as data:
        assert data["list"] == [1, 2, 3]
        assert data["dict"]["a"] == [4, 5, 6]


def test_empty_context():
    """Test that empty context manager works correctly."""
    dm = DataManager()

    with dm.access_data():
        pass

    assert isinstance(dm.get_data_snapshot(), dict)


def test_data_type_handling(data_manager, sample_data):
    """Test handling of various data types."""
    with data_manager.access_data() as data:
        data.update(sample_data)

    snapshot = data_manager.get_data_snapshot()
    for key, value in sample_data.items():
        assert snapshot.get(key) == value
