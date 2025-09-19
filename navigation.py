#!/usr/bin/env python3
"""
Navigation Module for Vision AI Assistant
Uses OpenRouteService API to provide turn-by-turn directions
"""

import requests
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from voice_output import VoiceOutput, speak


class NavigationAssistant:
    """Navigation assistant using OpenRouteService API"""
    
    def __init__(self, api_key: str, step_length: float = 0.75):
        """
        Initialize navigation assistant
        
        Args:
            api_key: OpenRouteService API key
            step_length: Length of one step in meters (default 0.75m)
        """
        self.api_key = api_key
        self.step_length = step_length
        self.base_url = "https://api.openrouteservice.org/v2"
        self.voice = VoiceOutput()
        
        # Profile options for different travel modes
        self.profiles = {
            'walking': 'foot-walking',
            'cycling': 'cycling-regular',
            'driving': 'driving-car',
            'wheelchair': 'wheelchair'
        }
        
        print("✅ Navigation assistant initialized!")
    
    def meters_to_steps(self, meters: float) -> int:
        """Convert meters to steps"""
        return max(1, round(meters / self.step_length))
    
    def format_distance(self, meters: float) -> str:
        """Format distance in a human-readable way"""
        if meters < 1000:
            return f"{meters:.0f} meters"
        else:
            return f"{meters/1000:.2f} km"
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in a human-readable way"""
        minutes = round(seconds / 60, 1)
        if minutes < 60:
            return f"{minutes} minutes"
        else:
            hours = int(minutes // 60)
            remaining_minutes = int(minutes % 60)
            return f"{hours}h {remaining_minutes}m"
    
    def get_directions(self, 
                      start_coords: List[float], 
                      end_coords: List[float], 
                      profile: str = 'walking',
                      instructions: bool = True) -> Dict[str, Any]:
        """
        Get route directions from OpenRouteService API
        
        Args:
            start_coords: [longitude, latitude] of start point
            end_coords: [longitude, latitude] of end point
            profile: Travel profile ('walking', 'cycling', 'driving', 'wheelchair')
            instructions: Whether to include step-by-step instructions
            
        Returns:
            Dictionary with route information
        """
        if profile not in self.profiles:
            return {"error": f"Invalid profile. Choose from: {list(self.profiles.keys())}"}
        
        if not self.api_key or self.api_key == "YOUR_ORS_API_KEY":
            return {"error": "Please provide a valid OpenRouteService API key"}
        
        try:
            url = f"{self.base_url}/directions/{self.profiles[profile]}"
            headers = {
                "Authorization": self.api_key,
                "Content-Type": "application/json"
            }
            
            body = {
                "coordinates": [start_coords, end_coords],
                "instructions": instructions,
                "format": "json"
            }
            
            print(f"🗺️ Calculating route from {start_coords} to {end_coords}...")
            response = requests.post(url, headers=headers, json=body, timeout=10)
            
            if response.status_code != 200:
                error_msg = f"API Error {response.status_code}: {response.text}"
                print(f"❌ {error_msg}")
                return {"error": error_msg}
            
            data = response.json()
            
            if not data.get("features"):
                return {"error": "No route found between the specified locations"}
            
            route = data["features"][0]
            properties = route["properties"]
            segments = properties["segments"][0]
            
            # Extract route information
            distance_m = segments["distance"]
            duration_s = segments["duration"]
            steps = self.meters_to_steps(distance_m)
            
            # Extract instructions
            instruction_list = []
            if instructions and "steps" in segments:
                for step in segments["steps"]:
                    instruction = {
                        "instruction": step["instruction"],
                        "distance": step["distance"],
                        "duration": step["duration"],
                        "type": step.get("type", 0),
                        "name": step.get("name", "")
                    }
                    instruction_list.append(instruction)
            
            result = {
                "success": True,
                "distance_m": distance_m,
                "distance_formatted": self.format_distance(distance_m),
                "steps": steps,
                "duration_s": duration_s,
                "duration_formatted": self.format_duration(duration_s),
                "instructions": instruction_list,
                "profile": profile,
                "start_coords": start_coords,
                "end_coords": end_coords
            }
            
            print(f"✅ Route calculated: {result['distance_formatted']} ({steps} steps), {result['duration_formatted']}")
            return result
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
    
    def get_directions_by_address(self, 
                                 start_address: str, 
                                 end_address: str, 
                                 profile: str = 'walking') -> Dict[str, Any]:
        """
        Get directions using address strings (requires geocoding)
        
        Args:
            start_address: Starting address
            end_address: Destination address
            profile: Travel profile
            
        Returns:
            Dictionary with route information
        """
        # First, geocode the addresses to get coordinates
        start_coords = self.geocode_address(start_address)
        if "error" in start_coords:
            return start_coords
        
        end_coords = self.geocode_address(end_address)
        if "error" in end_coords:
            return end_coords
        
        # Get directions using coordinates
        return self.get_directions(start_coords, end_coords, profile)
    
    def geocode_address(self, address: str) -> Dict[str, Any]:
        """
        Geocode an address to get coordinates
        
        Args:
            address: Address string to geocode
            
        Returns:
            Dictionary with coordinates or error
        """
        try:
            url = f"{self.base_url}/geocode/search"
            headers = {
                "Authorization": self.api_key,
                "Content-Type": "application/json"
            }
            
            params = {
                "text": address,
                "size": 1
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code != 200:
                return {"error": f"Geocoding error: {response.text}"}
            
            data = response.json()
            
            if not data.get("features"):
                return {"error": f"Address not found: {address}"}
            
            coords = data["features"][0]["geometry"]["coordinates"]
            return {"success": True, "coordinates": coords}
            
        except Exception as e:
            return {"error": f"Geocoding error: {str(e)}"}
    
    def speak_directions(self, route_info: Dict[str, Any], step_by_step: bool = True):
        """
        Speak the route directions using voice output
        
        Args:
            route_info: Route information from get_directions
            step_by_step: Whether to speak each instruction
        """
        if "error" in route_info:
            self.voice.speak(f"Navigation error: {route_info['error']}")
            return
        
        # Speak route summary
        summary = (f"Route found. Distance: {route_info['distance_formatted']}, "
                  f"about {route_info['steps']} steps. "
                  f"Estimated time: {route_info['duration_formatted']}")
        
        self.voice.speak(summary)
        
        if step_by_step and route_info.get("instructions"):
            time.sleep(2)  # Pause between summary and instructions
            
            self.voice.speak("Here are the step-by-step directions:")
            
            for i, instruction in enumerate(route_info["instructions"][:5]):  # Limit to first 5 steps
                step_text = f"Step {i+1}: {instruction['instruction']}"
                if instruction.get("name"):
                    step_text += f" on {instruction['name']}"
                
                self.voice.speak(step_text)
                time.sleep(1)  # Pause between instructions
    
    def get_current_location(self) -> Optional[Tuple[float, float]]:
        """
        Get current location (placeholder - would integrate with GPS)
        
        Returns:
            Tuple of (longitude, latitude) or None
        """
        # This is a placeholder - in a real implementation, you would:
        # 1. Use device GPS
        # 2. Use IP-based location
        # 3. Ask user for current location
        print("📍 Current location not available. Please provide coordinates manually.")
        return None
    
    def format_instruction_for_voice(self, instruction: Dict[str, Any]) -> str:
        """
        Format a single instruction for voice output
        
        Args:
            instruction: Instruction dictionary from API
            
        Returns:
            Formatted instruction string
        """
        text = instruction["instruction"]
        distance = instruction.get("distance", 0)
        name = instruction.get("name", "")
        
        # Add distance information
        if distance > 0:
            distance_text = self.format_distance(distance)
            text = f"{text} for {distance_text}"
        
        # Add street name if available
        if name:
            text = f"{text} on {name}"
        
        return text
    
    def get_route_summary(self, route_info: Dict[str, Any]) -> str:
        """
        Get a text summary of the route
        
        Args:
            route_info: Route information from get_directions
            
        Returns:
            Formatted summary string
        """
        if "error" in route_info:
            return f"Navigation error: {route_info['error']}"
        
        summary = (f"Route Summary:\n"
                  f"Distance: {route_info['distance_formatted']} ({route_info['steps']} steps)\n"
                  f"Duration: {route_info['duration_formatted']}\n"
                  f"Profile: {route_info['profile']}\n"
                  f"Instructions: {len(route_info.get('instructions', []))} steps")
        
        return summary


def main():
    """Example usage of the navigation module"""
    
    print("Navigation Module for Vision AI Assistant")
    print("=" * 50)
    
    # Get API key from user
    api_key = input("Enter your OpenRouteService API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("⚠️ No API key provided. Using demo mode.")
        api_key = "YOUR_ORS_API_KEY"
    
    # Initialize navigation assistant
    nav = NavigationAssistant(api_key)
    
    # Example coordinates (Chennai, India)
    start_coords = [79.1378, 12.9451]  # [longitude, latitude]
    end_coords = [79.1502, 12.9505]
    
    print(f"\n🗺️ Getting directions from {start_coords} to {end_coords}")
    
    # Get directions
    route = nav.get_directions(start_coords, end_coords)
    
    if "error" in route:
        print(f"❌ Error: {route['error']}")
        return
    
    # Print route information
    print("\n" + "=" * 50)
    print(nav.get_route_summary(route))
    print("=" * 50)
    
    # Show first few instructions
    if route.get("instructions"):
        print("\n📋 Step-by-step directions:")
        for i, instruction in enumerate(route["instructions"][:5]):
            formatted = nav.format_instruction_for_voice(instruction)
            print(f"{i+1}. {formatted}")
    
    # Ask if user wants voice output
    if nav.voice.is_available():
        voice_choice = input("\n🎤 Speak directions? (y/n): ").lower().strip()
        if voice_choice == 'y':
            nav.speak_directions(route)
    
    print("\n✅ Navigation example completed!")


# Convenience functions for easy integration
def get_directions(start_coords: List[float], 
                  end_coords: List[float], 
                  api_key: str = None,
                  profile: str = 'walking') -> Dict[str, Any]:
    """
    Convenience function to get directions
    
    Args:
        start_coords: [longitude, latitude] of start point
        end_coords: [longitude, latitude] of end point
        api_key: OpenRouteService API key
        profile: Travel profile
        
    Returns:
        Route information dictionary
    """
    if not api_key:
        api_key = "YOUR_ORS_API_KEY"
    
    nav = NavigationAssistant(api_key)
    return nav.get_directions(start_coords, end_coords, profile)


def speak_directions(route_info: Dict[str, Any], api_key: str = None):
    """
    Convenience function to speak directions
    
    Args:
        route_info: Route information from get_directions
        api_key: OpenRouteService API key
    """
    if not api_key:
        api_key = "YOUR_ORS_API_KEY"
    
    nav = NavigationAssistant(api_key)
    nav.speak_directions(route_info)


if __name__ == "__main__":
    main()
