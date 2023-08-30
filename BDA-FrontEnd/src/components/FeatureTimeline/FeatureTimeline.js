import { useQuasar } from "quasar";
import { computed } from "vue";
import { defineComponent } from "vue";

export default defineComponent({
  name: "FeatureTimeline",
  data (){
    return{
      title_1: ["TRIP PLANNER FEATURES:","Your Complete Trip in One Place"],
      heading: ["Budget Based Recommendation","Integrated Maps", "Day and Night activities recommendation"
      , "Get Top Recommendations","Itinerary Planning"],
      content: ["A tool that simplifies trip planning and allows users to spend more time enjoying their destination. " +
      "Our budget-based trip planning feature reduces the time and effort required for trip planning.\n" +
      " The main aim of this user friendly and intuitive webapp is to alleviate the stress associated with vacation\n" +
      " planning and enhance the overall travel experience for our users based on their budget ranges.",
      "Our fun-activity recommendation system provides personalized recommendations for things to do throughout\n" +
      "            the day and night,\n" +
      "            based on the user's preferred time.\n" +
      "            By utilizing our fun-activity recommendation system, travelers can easily discover and plan their itinerary\n" +
      "            around their preferred day and night schedule, enhancing their overall travel experience.",
      "We provide personalized itinerary recommendations based on the user's interests and preferences, including\n" +
      "            both daytime and nighttime activities. Also get weather details for each place which will help in better planning."]

    }
  }
});


