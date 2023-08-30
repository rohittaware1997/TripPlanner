# BDA-BackEnd

POST MAN CALL => http://127.0.0.1:5001/categories/get_categories
REQUEST BODY =>
{
    "province": "british_columbia",
    "low": 115.0,
    "high": 205.0,
    "cat_rating": {"private_&_custom_tours": 4.0, "luxury_&_special_occasions": 3.0, "sightseeing_tickets_&_passes": 1.0, "multi-day_&_extended_tours": 3.0, "walking_&_biking_tours": 1.0},
    "begin_date": "2023-04-05",
    "end_date": "2023-04-06"   
}

RESPONSE OF API =>
{
  "categories": {
    "1": [
      {
        "category": "featured_tours_and_tickets",
        "location": [
          49.1978340149,
          -123.064994812
        ],
        "name": "whistler_small-group_day_trip_from_vancouver",
        "price": 145,
        "rating": 5,
        "timeofday": "Morning"
      },
      {
        "category": "tours_&_sightseeing",
        "location": [
          49.1978340149,
          -123.064994812
        ],
        "name": "vancouver_city_sightseeing_tour",
        "price": 80,
        "rating": 4.5,
        "timeofday": "Morning"
      },
      {
        "category": "recommended_experiences",
        "location": [
          50.1133308411,
          -122.9547576904
        ],
        "name": "whistler_sasquatch_zipline",
        "price": 135.45,
        "rating": 5,
        "timeofday": "Morning"
      },
      {
        "category": "walking_&_biking_tours",
        "location": [
          49.2859191895,
          -123.113822937
        ],
        "name": "little_italy_walking_food_tour_in_vancouver",
        "price": 73,
        "rating": 4.5,
        "timeofday": "Morning"
      },
      {
        "category": "cruises,_sailing_&_water_tours",
        "location": [
          48.4265937805,
          -123.3709182739
        ],
        "name": "summer_whale_watching_on_vancouver_island",
        "price": 145.95,
        "rating": 5,
        "timeofday": "Evening"
      },
      {
        "category": "recommended_experiences",
        "location": [
          49.7491493225,
          -123.1333770752
        ],
        "name": "overnight_camping_and_river-rafting_trip_in_squamish",
        "price": 425.24,
        "rating": 5,
        "timeofday": "Evening"
      },
      {
        "category": "recommended_experiences",
        "location": [
          49.2841453552,
          -123.1193161011
        ],
        "name": "vancouver_highlights_bike_tour",
        "price": 94.45,
        "rating": 5,
        "timeofday": "Evening"
      },
      {
        "category": "walking_&_biking_tours",
        "location": [
          49.2914733887,
          -123.1424026489
        ],
        "name": "vancouver_biking_and_hiking_tour_including_lunch",
        "price": 135.5,
        "rating": 5,
        "timeofday": "Evening"
      }
    ],
    "2": [
      {
        "category": "outdoor_activities",
        "location": [
          50.1133766174,
          -122.9542617798
        ],
        "name": "call_of_the_wild_atv_tour",
        "price": 156.45,
        "rating": 5,
        "timeofday": "Morning"
      },
      {
        "category": "cruises,_sailing_&_water_tours",
        "location": [
          48.422203064,
          -123.3796234131
        ],
        "name": "victoria_whale_watching_tour_on_a_covered_vessel",
        "price": 122,
        "rating": 4,
        "timeofday": "Morning"
      },
      {
        "category": "cruises,_sailing_&_water_tours",
        "location": [
          48.4227218628,
          -123.3689804077
        ],
        "name": "half-day_whale_watching_adventure_from_vancouver",
        "price": 181.13,
        "rating": 4.5,
        "timeofday": "Morning"
      },
      {
        "category": "water_sports",
        "location": [
          49.2717285156,
          -123.1366577148
        ],
        "name": "english_bay_jet_ski_tour_from_vancouver",
        "price": 155.68,
        "rating": 5,
        "timeofday": "Morning"
      },
      {
        "category": "walking_&_biking_tours",
        "location": [
          48.4300460815,
          -123.3637237549
        ],
        "name": "victoria_guided_food_and_history_tour",
        "price": 63,
        "rating": 5,
        "timeofday": "Evening"
      },
      {
        "category": "water_sports",
        "location": [
          49.1245117188,
          -123.1840057373
        ],
        "name": "whale-watching_tour_from_vancouver",
        "price": 138.5,
        "rating": 4.5,
        "timeofday": "Evening"
      },
      {
        "category": "recommended_experiences",
        "location": [
          48.4224090576,
          -123.3670501709
        ],
        "name": "whale-watching_cruise_with_expert_naturalists",
        "price": 94.5,
        "rating": 5,
        "timeofday": "Evening"
      },
      {
        "category": "day_trips_&_excursions",
        "location": [
          49.1978340149,
          -123.064994812
        ],
        "name": "vancouver_to_victoria_and_butchart_gardens_tour_by_bus",
        "price": 210,
        "rating": 5,
        "timeofday": "Evening"
      }
    ]
  },
  "recommandations": {
    "1": [
      {
        "category": "walking_&_biking_tours",
        "location": [
          49.2859191895,
          -123.113822937
        ],
        "name": "little_italy_walking_food_tour_in_vancouver",
        "price": 73,
        "rating": 4.5,
        "timeofday": "Morning"
      },
      {
        "category": "tours_&_sightseeing",
        "location": [
          49.1978340149,
          -123.064994812
        ],
        "name": "vancouver_city_sightseeing_tour",
        "price": 80,
        "rating": 4.5,
        "timeofday": "Morning"
      },
      {
        "category": "recommended_experiences",
        "location": [
          49.2841453552,
          -123.1193161011
        ],
        "name": "vancouver_highlights_bike_tour",
        "price": 94.45,
        "rating": 5,
        "timeofday": "Evening"
      },
      {
        "category": "walking_&_biking_tours",
        "location": [
          49.2914733887,
          -123.1424026489
        ],
        "name": "vancouver_biking_and_hiking_tour_including_lunch",
        "price": 135.5,
        "rating": 5,
        "timeofday": "Evening"
      }
    ],
    "2": [
      {
        "category": "cruises,_sailing_&_water_tours",
        "location": [
          48.422203064,
          -123.3796234131
        ],
        "name": "victoria_whale_watching_tour_on_a_covered_vessel",
        "price": 122,
        "rating": 4,
        "timeofday": "Morning"
      },
      {
        "category": "water_sports",
        "location": [
          49.2717285156,
          -123.1366577148
        ],
        "name": "english_bay_jet_ski_tour_from_vancouver",
        "price": 155.68,
        "rating": 5,
        "timeofday": "Morning"
      },
      {
        "category": "walking_&_biking_tours",
        "location": [
          48.4300460815,
          -123.3637237549
        ],
        "name": "victoria_guided_food_and_history_tour",
        "price": 63,
        "rating": 5,
        "timeofday": "Evening"
      },
      {
        "category": "recommended_experiences",
        "location": [
          48.4224090576,
          -123.3670501709
        ],
        "name": "whale-watching_cruise_with_expert_naturalists",
        "price": 94.5,
        "rating": 5,
        "timeofday": "Evening"
      }
    ]
  }
}
