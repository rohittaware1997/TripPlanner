import { defineComponent, ref } from "vue";
import FeatureTimeline from "components/FeatureTimeline/FeatureTimeline.vue";
import SearchBar from "components/SearchBar/SearchBar.vue";
import GoogleMap from "components/GoogleMap/GoogleMap.vue";
import Weather from "components/Weather/WeatherAPI.vue";
import TileSelection from "components/TileSelection/TileSelection.vue";
import ParallaxScroll from "components/ParallaxScroll/ParallaxScroll.vue";
import { EventBus } from "quasar";


const bus = new EventBus();

export default defineComponent({
  name: "LandingPage",
  components: {
    Weather,
    GoogleMap,
    FeatureTimeline,
    SearchBar,
    TileSelection,
    ParallaxScroll
  },
  setup() {
    const attraction_types = ref([
      {
        id: 0,
        display_name: "PRIVATE & CUSTOM TOURS",
        actual_name: "private_&_custom_tours",
        disabled: false
      },
      {
        id: 1,
        display_name: "SIGHTSEEING TOURS",
        actual_name: "tours_&_sightseeing",
        disabled: false
      },
      {
        id: 2,
        display_name: "RECOMMENDED EXPERIENCES",
        actual_name: "recommended_experiences",
        disabled: false
      },
      {
        id: 3,
        display_name: "OUTDOOR ACTIVITIES",
        actual_name: "outdoor_activities",
        disabled: false
      },
      {
        id: 4,
        display_name: "FOOD, WINE & NIGHTLIFE",
        actual_name: "food,_wine_&_nightlife",
        disabled: false
      },
      {
        id: 5,
        display_name: "FAMILY FRIENDLY",
        actual_name: "family_friendly",
        disabled: false
      }, {
        id: 6,
        display_name: "CULTURAL & THEME TOURS",
        actual_name: "cultural_&_theme_tours",
        disabled: false
      },
      {
        id: 7,
        display_name: "WALKING & BIKING TOURS",
        actual_name: "walking_&_biking_tours",
        disabled: false
      },
      {
        id: 8,
        display_name: "DAY TRIPS & EXCURSIONS",
        actual_name: "day_trips_&_excursions",
        disabled: false
      }, {
        id: 9,
        display_name: "WATER SPORTS",
        actual_name: "water_sports",
        disabled: false
      }, {
        id: 10,
        display_name: "CRUISES, SAILING & WATER TOURS",
        actual_name: "cruises,_sailing_&_water_tours",
        disabled: false
      }, {
        id: 11,
        display_name: "TRANSFERS & GROUND TRANSPORT",
        actual_name: "transfers_&_ground_transport",
        disabled: false
      },
      {
        id: 12,
        display_name: "LUXURY & SPECIAL OCCASIONS",
        actual_name: "luxury_&_special_occasions",
        disabled: false
      },
      {
        id: 13,
        display_name: "SIGHTSEEING, TICKETS & PASSES",
        actual_name: "sightseeing_tickets_&_passes",
        disabled: false
      },
      {
        id: 14,
        display_name: "MULTI-DAY & EXTENDED TOURS",
        actual_name: "multi-day_&_extended_tours",
        disabled: false
      },
      {
        id: 15,
        display_name: "SHORE EXCURSIONS",
        actual_name: "shore_excursions",
        disabled: false
      },
      {
        id: 16,
        display_name: "AIR, HELICOPTER & BALLOON TOURS",
        actual_name: "air,_helicopter_&_balloon_tours",
        disabled: false
      },
      {
        id: 17,
        display_name: "HOLIDAY & SEASONAL TOURS",
        actual_name: "holiday_&_seasonal_tours",
        disabled: false
      },
      {
        id: 18,
        display_name: "SHOWS, CONCERTS & SPORTS",
        actual_name: "shows,_concerts_&_sports",
        disabled: false
      },
      {
        id: 19,
        display_name: "CLASSES & WORKSHOPS",
        actual_name: "classes_&_workshops",
        disabled: false
      }

    ]);
    const hotel_types = ref([
      {
        id: 0,
        display_name: "NONSMOKING HOTEL",
        actual_name: " Nonsmoking hotel",
        disabled: false
      },
      {
        id: 1,
        display_name: "LAUNDRY SERVICE",
        actual_name: " Laundry Service",
        disabled: false
      },
      {
        id: 2,
        display_name: "FREE HIGH SPEED INTERNET WIFI",
        actual_name: " Free High Speed Internet WiFi",
        disabled: false
      },
      {
        id: 3,
        display_name: "AIR CONDITIONING",
        actual_name: " Air conditioning",
        disabled: false
      },
      {
        id: 4,
        display_name: "REFRIGERATOR IN ROOM",
        actual_name: " Refrigerator in room",
        disabled: false
      },
      {
        id: 5,
        display_name: "FAMILY ROOMS",
        actual_name: " Family Rooms",
        disabled: false
      },
      {
        id: 6,
        display_name: "PUBLIC WIFI",
        actual_name: " Public Wifi",
        disabled: false
      },
      {
        id: 7,
        display_name: "PETS ALLOWED & DOG FRIENDLY",
        actual_name: " Pets Allowed  Dog  Pet Friendly ",
        disabled: false
      },
      {
        id: 8,
        display_name: "SUITES",
        actual_name: " Suites",
        disabled: false
      },
      {
        id: 9,
        display_name: "WHEELCHAIR ACCESS",
        actual_name: " Wheelchair Access",
        disabled: false
      },
      {
        id: 10,
        display_name: "MICROWAVE",
        actual_name: " Microwave",
        disabled: false
      },
      {
        id: 11,
        display_name: "BREAKFAST INCLUDED",
        actual_name: " Breakfast included",
        disabled: false
      }
    ]);
    const nextParam = ref(true);
    return {
      step: ref(1),
      attraction_types,
      nextParam,
      hotel_types
    };
  },

  methods: {
    handleValueChanged(value) {
      this.nextParam = value;
    },
    jumpToElement(element_id) {
      let element = document.getElementById(element_id);
      element.scrollIntoView({ behavior: "smooth" });
    },
    triggerFunction() {
      this.$emit("secondPage", true);
    }
  }
});
