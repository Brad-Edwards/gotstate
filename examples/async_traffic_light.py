#!/usr/bin/env python3
"""
Async Traffic Light Example

This example demonstrates the usage of gotstate's AsyncStateMachine for implementing 
a traffic light state machine with three states: red, yellow, and green.
"""

import asyncio
from gotstate import State
from gotstate.extensions.async_sm import AsyncStateMachine


async def create_traffic_light() -> AsyncStateMachine:
    """Create a traffic light state machine."""
    # Create an async state machine
    machine = AsyncStateMachine("traffic_light")
    
    # Define states
    red = State("red")
    yellow = State("yellow")
    green = State("green")
    
    # Add states to the machine
    machine.add_state(red, initial=True)
    machine.add_state(yellow)
    machine.add_state(green)
    
    # Add transitions
    machine.add_transition(red, green, "timer_expired")
    machine.add_transition(green, yellow, "timer_expired")
    machine.add_transition(yellow, red, "timer_expired")
    
    # Add entry actions
    @red.on_entry
    def red_light_on(event_id, event_data):
        print("\nRed light on. Stop!")
    
    @yellow.on_entry
    def yellow_light_on(event_id, event_data):
        print("\nYellow light on. Prepare to stop!")
    
    @green.on_entry
    def green_light_on(event_id, event_data):
        print("\nGreen light on. Go!")
    
    return machine


async def run_traffic_light(machine: AsyncStateMachine, cycles: int = 3, interval: float = 2.0) -> None:
    """
    Run the traffic light for a number of cycles asynchronously.
    
    Args:
        machine: The async state machine to run
        cycles: Number of full cycles to run
        interval: Time interval in seconds between transitions
    """
    print("Starting traffic light...")
    await machine.start()
    
    # Start the event processing loop in the background
    event_loop_task = asyncio.create_task(machine.run_event_loop())
    
    try:
        # Run for the specified number of cycles
        for cycle in range(1, cycles + 1):
            print(f"\nCycle {cycle} of {cycles}")
            
            for _ in range(3):  # 3 lights in one cycle
                await asyncio.sleep(interval)
                print("Timer expired. Transitioning...")
                await machine.process_event("timer_expired")
        
        print("\nTraffic light operation complete.")
    finally:
        # Stop the state machine and cancel the event loop task
        await machine.stop()
        event_loop_task.cancel()
        try:
            await event_loop_task
        except asyncio.CancelledError:
            pass


async def main():
    """Main async function."""
    traffic_light = await create_traffic_light()
    await run_traffic_light(traffic_light)


if __name__ == "__main__":
    asyncio.run(main()) 