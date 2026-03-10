"""Tests for gotstate.extensions modules (hooks, sandbox)."""

import pytest

from gotstate.exceptions import GotStateError
from gotstate.extensions.hooks import (
    ExtensionHooks,
    HookManager,
    HookPhase,
    HookPriority,
)
from gotstate.extensions.sandbox import (
    ExtensionSandbox,
    ResourceLimit,
    SecurityLevel,
)


class ConcreteHook(ExtensionHooks):
    """Test hook implementation."""

    def __init__(self, name, priority=HookPriority.NORMAL):
        super().__init__(name, priority)
        self.pre_calls = []
        self.post_calls = []
        self.error_calls = []

    def on_pre(self, context):
        self.pre_calls.append(context)

    def on_post(self, context):
        self.post_calls.append(context)

    def on_error(self, context, error):
        self.error_calls.append((context, error))


class TestExtensionHooks:
    def test_create(self):
        hook = ConcreteHook("test")
        assert hook.name == "test"
        assert hook.priority == HookPriority.NORMAL
        assert not hook.is_active

    def test_activate_deactivate(self):
        hook = ConcreteHook("test")
        hook.activate()
        assert hook.is_active
        hook.deactivate()
        assert not hook.is_active


class TestHookManager:
    def test_register_and_execute(self):
        mgr = HookManager()
        hook = ConcreteHook("test")
        mgr.register("transition_pre", hook)
        assert hook.is_active

        mgr.execute_hooks("transition_pre", {"state": "idle"})
        assert len(hook.pre_calls) == 1
        assert hook.pre_calls[0] == {"state": "idle"}

    def test_execute_post(self):
        mgr = HookManager()
        hook = ConcreteHook("test")
        mgr.register("transition_post", hook)
        mgr.execute_hooks("transition_post", {"state": "active"})
        assert len(hook.post_calls) == 1

    def test_unregister(self):
        mgr = HookManager()
        hook = ConcreteHook("test")
        mgr.register("phase", hook)
        mgr.unregister("phase", "test")
        assert mgr.get_hooks("phase") == []

    def test_priority_ordering(self):
        mgr = HookManager()
        h1 = ConcreteHook("low", HookPriority.LOW)
        h2 = ConcreteHook("high", HookPriority.HIGH)
        mgr.register("phase_pre", h1)
        mgr.register("phase_pre", h2)
        hooks = mgr.get_hooks("phase_pre")
        assert hooks[0].name == "high"
        assert hooks[1].name == "low"

    def test_inactive_hooks_not_executed(self):
        mgr = HookManager()
        hook = ConcreteHook("test")
        mgr.register("phase_pre", hook)
        hook.deactivate()
        mgr.execute_hooks("phase_pre", {})
        assert len(hook.pre_calls) == 0

    def test_empty_phase_no_error(self):
        mgr = HookManager()
        mgr.execute_hooks("nonexistent", {})  # Should not raise


class TestExtensionSandbox:
    def test_create(self):
        sb = ExtensionSandbox()
        assert sb.security_level == SecurityLevel.STANDARD
        assert sb.violations == []

    def test_create_with_level(self):
        sb = ExtensionSandbox(SecurityLevel.STRICT)
        assert sb.security_level == SecurityLevel.STRICT

    def test_resource_limits(self):
        sb = ExtensionSandbox()
        sb.set_resource_limit(ResourceLimit.MEMORY, 100.0)
        assert sb.check_resource(ResourceLimit.MEMORY, 50.0)
        assert not sb.check_resource(ResourceLimit.MEMORY, 150.0)

    def test_record_usage_within_limit(self):
        sb = ExtensionSandbox()
        sb.set_resource_limit(ResourceLimit.MEMORY, 100.0)
        sb.record_usage(ResourceLimit.MEMORY, 50.0)
        sb.record_usage(ResourceLimit.MEMORY, 30.0)
        # 80 total, still within 100

    def test_record_usage_exceeds_limit(self):
        sb = ExtensionSandbox()
        sb.set_resource_limit(ResourceLimit.MEMORY, 100.0)
        sb.record_usage(ResourceLimit.MEMORY, 80.0)
        with pytest.raises(GotStateError, match="Resource limit exceeded"):
            sb.record_usage(ResourceLimit.MEMORY, 30.0)
        assert len(sb.violations) == 1

    def test_record_usage_no_limit_set(self):
        sb = ExtensionSandbox()
        sb.record_usage(ResourceLimit.CPU, 999.0)  # No limit, should not raise

    def test_operation_allowed_standard(self):
        sb = ExtensionSandbox(SecurityLevel.STANDARD)
        assert not sb.is_operation_allowed("read")
        sb.allow_operation("read")
        assert sb.is_operation_allowed("read")
        assert not sb.is_operation_allowed("write")

    def test_operation_allowed_relaxed(self):
        sb = ExtensionSandbox(SecurityLevel.RELAXED)
        assert sb.is_operation_allowed("anything")

    def test_execute_sandboxed(self):
        sb = ExtensionSandbox()
        result = sb.execute_sandboxed(lambda: 42)
        assert result == 42

    def test_execute_sandboxed_error_recorded(self):
        sb = ExtensionSandbox()
        with pytest.raises(ValueError):
            sb.execute_sandboxed(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert len(sb.violations) == 1
        assert sb.violations[0]["type"] == "execution_error"

    def test_reset_usage(self):
        sb = ExtensionSandbox()
        sb.set_resource_limit(ResourceLimit.MEMORY, 100.0)
        sb.record_usage(ResourceLimit.MEMORY, 90.0)
        sb.reset_usage()
        sb.record_usage(ResourceLimit.MEMORY, 90.0)  # Should work after reset
