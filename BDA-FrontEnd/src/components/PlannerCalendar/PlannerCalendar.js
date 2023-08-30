import { defineComponent } from "vue";
import VueCal from "vue-cal";
import "vue-cal/dist/vuecal.css";
import { useStore } from "vuex";

export default defineComponent({
  name: "PlannerCalendar",
  components: { VueCal },
  props: ["events", "height","sDate"],
  setup() {
    const store = useStore();
    return {
      store
    };
  },
  methods: {
    delEvent (emittedEventName, params) {
      let budgetString = params.content;
      let budNumber = parseFloat(budgetString.match(/\d+(\.\d+)?/)[0]);
      this.$emit("removeFromCal", params.title, budNumber);
      this.store.dispatch("planner/updateToggleCal", params.title);
    },
  }
});
