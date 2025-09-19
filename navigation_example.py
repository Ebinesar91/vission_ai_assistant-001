#!/usr/bin/env python3
"""
Navigation Example - Test OpenRouteService Integration
Demonstrates how to use the navigation module
"""

from navigation import NavigationAssistant, get_directions, speak_directions
from voice_output import VoiceOutput


def test_navigation():
    """Test navigation functionality"""
    
    print("Navigation Module Test")
    print("=" * 40)
    
    # Get API key from user
    api_key = input("Enter your OpenRouteService API key (or press Enter for demo): ").strip()
    
    if not api_key:
        print("⚠️ No API key provided. Using demo mode with sample data.")
        api_key = "YOUR_ORS_API_KEY"
    
    # Initialize navigation assistant
    nav = NavigationAssistant(api_key)
    
    # Test coordinates (Chennai, India)
    start_coords = [79.1378, 12.9451]  # [longitude, latitude]
    end_coords = [79.1502, 12.9505]
    
    print(f"\n🗺️ Testing navigation from {start_coords} to {end_coords}")
    
    # Test 1: Basic directions
    print("\n1. Testing basic directions...")
    route = nav.get_directions(start_coords, end_coords)
    
    if "error" in route:
        print(f"❌ Error: {route['error']}")
        print("This is expected without a valid API key.")
        print("\nTo get real directions:")
        print("1. Sign up at https://openrouteservice.org/sign-up/")
        print("2. Get your free API key")
        print("3. Run this script with your API key")
        return
    
    # Print route information
    print("✅ Route calculated successfully!")
    print(f"Distance: {route['distance_formatted']} ({route['steps']} steps)")
    print(f"Duration: {route['duration_formatted']}")
    print(f"Instructions: {len(route['instructions'])} steps")
    
    # Test 2: Voice output
    print("\n2. Testing voice output...")
    if nav.voice.is_available():
        voice_choice = input("Speak directions? (y/n): ").lower().strip()
        if voice_choice == 'y':
            nav.speak_directions(route)
    else:
        print("❌ Voice output not available")
    
    # Test 3: Step-by-step instructions
    print("\n3. Step-by-step instructions:")
    if route.get("instructions"):
        for i, instruction in enumerate(route["instructions"][:5]):
            formatted = nav.format_instruction_for_voice(instruction)
            print(f"{i+1}. {formatted}")
    
    # Test 4: Different travel profiles
    print("\n4. Testing different travel profiles...")
    profiles = ['walking', 'cycling', 'driving']
    
    for profile in profiles:
        print(f"\nTesting {profile} profile...")
        route_profile = nav.get_directions(start_coords, end_coords, profile)
        
        if "error" not in route_profile:
            print(f"  Distance: {route_profile['distance_formatted']}")
            print(f"  Duration: {route_profile['duration_formatted']}")
        else:
            print(f"  Error: {route_profile['error']}")
    
    print("\n✅ Navigation test completed!")


def test_convenience_functions():
    """Test convenience functions"""
    
    print("\n" + "=" * 40)
    print("Testing Convenience Functions")
    print("=" * 40)
    
    # Test convenience function
    start = [79.1378, 12.9451]
    end = [79.1502, 12.9505]
    
    print(f"Testing get_directions({start}, {end})...")
    route = get_directions(start, end)
    
    if "error" in route:
        print(f"❌ Error: {route['error']}")
    else:
        print(f"✅ Success: {route['distance_formatted']}, {route['duration_formatted']}")
        
        # Test voice output
        if input("Test voice output? (y/n): ").lower().strip() == 'y':
            speak_directions(route)


def demo_without_api():
    """Demo the navigation module without API key"""
    
    print("\n" + "=" * 40)
    print("Demo Mode - Navigation Module Features")
    print("=" * 40)
    
    # Initialize with demo API key
    nav = NavigationAssistant("YOUR_ORS_API_KEY")
    
    print("✅ Navigation assistant initialized")
    print("✅ Voice output available:", nav.voice.is_available())
    print("✅ Step length:", nav.step_length, "meters")
    print("✅ Available profiles:", list(nav.profiles.keys()))
    
    # Test utility functions
    print("\nTesting utility functions:")
    print(f"100 meters = {nav.meters_to_steps(100)} steps")
    print(f"Distance formatting: {nav.format_distance(1500)}")
    print(f"Duration formatting: {nav.format_duration(3661)}")
    
    # Test voice output
    if nav.voice.is_available():
        print("\nTesting voice output...")
        nav.voice.speak("Navigation module is ready for use!")
        nav.voice.speak("To get real directions, please provide a valid OpenRouteService API key.")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    print("OpenRouteService Navigation Module - Test Suite")
    print("=" * 60)
    
    # Run tests
    test_navigation()
    test_convenience_functions()
    demo_without_api()
    
    print("\n🎉 All tests completed!")
    print("\nTo use with real navigation:")
    print("1. Get API key from https://openrouteservice.org/sign-up/")
    print("2. Run: python navigation_example.py")
    print("3. Enter your API key when prompted")
