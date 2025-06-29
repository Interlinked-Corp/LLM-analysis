from typing import Dict, List
from bespokelabs import curator
from pydantic import BaseModel, Field
import os
import getpass
import random
import uuid
from dotenv import load_dotenv
import json

load_dotenv()

# Set API key for Anthropic (commented out for security; user should input their own)
# os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("sk-ant-api03-...")
# Enable curator viewer if desired
os.environ["HOSTED_CURATOR_VIEWER"] = "1"

class WildfireReport(BaseModel):
    precedent: Dict = Field(description="Details about the fire incident")
    weather_report: Dict = Field(description="Meteorological conditions affecting the fire")
    topography_report: Dict = Field(description="Terrain and fuel characteristics")
    fire_spread_predictions: Dict = Field(description="Predicted fire behavior and spread")
    land_of_interest: Dict = Field(description="Key areas and infrastructure at risk")
    resources: Dict = Field(description="Available resources for response")
    output: Dict = Field(description="Analysis, predictions, and recommendations with strategy generation")

class WildfireDataGenerator(curator.LLM):
    response_format = WildfireReport

    def prompt(self, input: str) -> str:
        return (
            "Generate a synthetic wildfire incident report for a hypothetical fire in a western U.S. state (e.g., Nevada, Utah, California). "
            "The report must follow the exact JSON structure provided below, treating 'precedent', 'weather_report', 'topography_report', 'fire_spread_predictions', 'land_of_interest', and 'resources' as input sections, and generating an 'output' section with exactly seven components: 'event_summary', 'detailed_analysis', 'predictions', 'impacts', 'recommendations', 'response_strategy', and 'evacuation_strategy'. "
            "Use decimal degrees for coordinates in the format 'XX.XXXX°N, -YYY.YYYY°W' (e.g., '39.4215°N, -114.8973°W'), 3-digit wind directions (degrees from North, e.g., '235.000'), 24-hour time format (e.g., '15:30'), and precise units (e.g., miles, acres/hour, kg/acre, feet, kts, °F, %). "
            "Base the data on typical wildfire scenarios in shrubland or forested regions, considering historical patterns (e.g., 2018 Martin Fire, 2017 Brian Head Fire). "
            "Populate 2-3 examples per iterative field (e.g., spread vectors, hotspots, access routes). "
            "Ensure all values are realistic, domain-appropriate, and consistent with firefighter vernacular, with no placeholders. "
            "The output must be valid JSON, with all strings properly enclosed in quotes, no trailing commas, and correct formatting for coordinates and numeric values. "
            "The output must be suitable for a government or agency wildfire situational briefing. "
            "For the 'output' section, follow these steps:\n"
            "1. Extract key data points from each input section.\n"
            "2. Summarize the event (type, location, status).\n"
            "3. Analyze the current situation, potential progression, and resulting events.\n"
            "4. Consider historical patterns (e.g., 2018 Martin Fire, 2017 Brian Head Fire).\n"
            "5. Predict trajectory, severity, duration, and key features.\n"
            "6. Assess impacts on urban areas, wildlife, and infrastructure.\n"
            "7. Estimate severity using historical patterns and current data.\n"
            "8. Generate actionable safety recommendations and mitigation strategies.\n"
            "9. Integrate weather context and its influence on fire behavior.\n"
            "10. Evaluate data reliability and completeness.\n"
            "11. Provide historical comparisons to similar and regional past events.\n"
            "12. Offer localized recommendations and critical observations.\n"
            "13. Quantify uncertainty with confidence metrics for predictions.\n"
            "Structure the output as a JSON object matching the following schema:\n"
            "{\n"
            "  \"instruction\": {\n"
            "    \"precedent\": {\n"
            "      \"incident_name\": str,\n"
            "      \"ignition_coordinates\": str,\n"
            "      \"current_radius_miles\": float,\n"
            "      \"maximum_predicted_radius_miles\": float,\n"
            "      \"quadrants\": Dict[str, str],\n"
            "      \"time_since_ignition\": str,\n"
            "      \"growth_rate_acres_per_hour\": int,\n"
            "      \"burning_index\": str,\n"
            "      \"containment_status\": str\n"
            "    },\n"
            "    \"weather_report\": {\n"
            "      \"temperature_f\": int,\n"
            "      \"low_temp\": int,\n"
            "      \"high_temp_time\": str,\n"
            "      \"low_temp_time\": str,\n"
            "      \"high_surface_temp_direct\": int,\n"
            "      \"high_surface_temp_direct_time\": str,\n"
            "      \"low_surface_temp_direct\": int,\n"
            "      \"low_surface_temp_direct_time\": str,\n"
            "      \"high_surface_temp_shade\": int,\n"
            "      \"high_surface_temp_shade_time\": str,\n"
            "      \"low_surface_temp_shade\": int,\n"
            "      \"low_surface_temp_shade_time\": str,\n"
            "      \"humidity_percent\": int,\n"
            "      \"humidity_high\": int,\n"
            "      \"humidity_high_time\": str,\n"
            "      \"humidity_low_time\": str,\n"
            "      \"wind_direction_day\": str,\n"
            "      \"wind_speed_kts_day\": int,\n"
            "      \"wind_direction_night\": str,\n"
            "      \"wind_speed_kts_night\": int,\n"
            "      \"forecast\": str,\n"
            "      \"wetting_rain_chance\": str,\n"
            "      \"precipitation_amt\": float,\n"
            "      \"cloud_cover_value\": str\n"
            "    },\n"
            "    \"topography_report\": {\n"
            "      \"terrain_type\": str,\n"
            "      \"fuel_type\": str,\n"
            "      \"low_fuel_density\": int,\n"
            "      \"low_fuel_coords\": str,\n"
            "      \"high_fuel_density\": int,\n"
            "      \"high_fuel_coords\": str,\n"
            "      \"avg_fuel_density\": int,\n"
            "      \"elevation\": {\n"
            "        \"lowest_elevation\": int,\n"
            "        \"lowest_elevation_coords\": str,\n"
            "        \"highest_elevation\": int,\n"
            "        \"highest_elevation_coords\": str\n"
            "      }\n"
            "    },\n"
            "    \"fire_spread_predictions\": {\n"
            "      \"mpr_zones\": List[str],\n"
            "      \"mpr_slopes\": List[str],\n"
            "      \"mpr_aspect_grads\": List[int],\n"
            "      \"spread_speeds\": List[int],\n"
            "      \"spread_speed_directions\": List[str],\n"
            "      \"spread_vectors\": List[str],\n"
            "      \"spread_vector_coords\": List[str],\n"
            "      \"spread_vector_causes\": List[str],\n"
            "      \"spread_consensus\": str,\n"
            "      \"spread_hotspots\": List[str],\n"
            "      \"spread_hotspot_elevations\": List[int],\n"
            "      \"spread_hotspot_intensities\": List[int],\n"
            "      \"spread_potential\": int,\n"
            "      \"spread_distance\": float,\n"
            "      \"spread_angle\": int\n"
            "    },\n"
            "    \"land_of_interest\": {\n"
            "      \"access_routes\": List[str],\n"
            "      \"access_coords\": List[str],\n"
            "      \"natural_barriers\": List[str],\n"
            "      \"barrier_coords\": List[str],\n"
            "      \"high_risk_area_types\": List[str],\n"
            "      \"high_risk_area_type_coords\": List[str],\n"
            "      \"ownership_coords\": List[str],\n"
            "      \"ownership_types\": List[str],\n"
            "      \"ownership_radii\": List[float],\n"
            "      \"wui_zones\": {\n"
            "        \"wui_community_names\": List[str],\n"
            "        \"wui_population_estimates\": List[int],\n"
            "        \"wui_impact_times\": List[int]\n"
            "      },\n"
            "      \"critical_infrastructure\": {\n"
            "        \"critical_infrastructures\": List[str],\n"
            "        \"critical_infrastructure_coords\": List[str]\n"
            "      },\n"
            "      \"protected_areas\": {\n"
            "        \"protected_areas\": List[str],\n"
            "        \"protected_area_radii\": List[float],\n"
            "        \"protected_area_coords\": List[str]\n"
            "      }\n"
            "    },\n"
            "    \"resources\": {\n"
            "      \"watershed_resources\": List[str],\n"
            "      \"watershed_resource_coords\": List[str],\n"
            "      \"available_resources\": List[str],\n"
            "      \"available_resources_details\": List[str]\n"
            "    },\n"
            "    \"output\": {\n"
            "      \"event_summary\": str,\n"
            "      \"detailed_analysis\": str,\n"
            "      \"predictions\": str,\n"
            "      \"impacts\": str,\n"
            "      \"recommendations\": str,\n"
            "      \"response_strategy\": str,\n"
            "      \"evacuation_strategy\": str\n"
            "    }\n"
            "  }\n"
            "}\n"
            "Ensure all values are realistic, based on typical wildfire scenarios in the western U.S., and consistent with historical fire behavior (e.g., rapid spread in shrublands, influence of wind and topography). "
            "Generate one complete wildfire report with all fields populated, including the 'output' section with only the seven specified components. "
            "Ensure the response is valid JSON, with proper coordinate formatting (e.g., '39.4215°N, -114.8973°W'), no unclosed quotes, and no trailing commas."
        )

    def parse(self, input: str, response: WildfireReport) -> Dict:
        return {
            "instruction": response.dict()
        }

# Initialize the generator
model_name = "claude-3-7-sonnet-20250219"
wildfire_generator = WildfireDataGenerator(model_name=model_name)

# Generate synthetic wildfire data
wildfire_data = wildfire_generator("wildfire incident")

# Write the result to a JSON file
with open("./wildfire_report.json", "w") as f:
    json.dump(wildfire_data, f, indent=2)