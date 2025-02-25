#!/usr/bin/env python3
"""
Traffic Light Example

This example demonstrates the usage of gotstate for implementing a traffic light
state machine with three states: red, yellow, and green.
"""

import time
from gotstate import StateMachine, State


def create_traffic_light() -> StateMachine:
    """Create a traffic light state machine."""
    # Create a state machine
    machine = StateMachine("traffic_light")
    
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


def run_traffic_light(machine: StateMachine, cycles: int = 3, interval: float = 2.0) -> None:
    """
    Run the traffic light for a number of cycles.
    
    Args:
        machine: The state machine to run
        cycles: Number of full cycles to run
        interval: Time interval in seconds between transitions
    """
    print("Starting traffic light...")
    machine.start()
    
    # Run for the specified number of cycles
    for cycle in range(1, cycles + 1):
        print(f"\nCycle {cycle} of {cycles}")
        
        for _ in range(3):  # 3 lights in one cycle
            time.sleep(interval)
            print("Timer expired. Transitioning...")
            machine.process_event("timer_expired")
    
    print("\nTraffic light operation complete.")


if __name__ == "__main__":
    traffic_light = create_traffic_light()
    run_traffic_light(traffic_light) 