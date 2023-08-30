import { computed, defineComponent } from "vue";
import { ref } from "vue";
import { useStore } from "vuex";

export default defineComponent({
  name: "SearchBar",

  beforeUnmount() {
    this.arrowButton();
  },
  setup() {

    const store = useStore();
    let range = ref({
      min: 100,
      max: 500
    });
    return {
      store,
      range,
      budget: ref(),
      date: ref(""),
      date2: ref(""),
      model: ref(""),
      amount: ref(),
      options: ref([
        {
          label: "Alberta", value: "alberta"
        },
        {
          label: "British Columbia", value: "british_columbia"
        },
        {
          label: "Northwest Territories", value: "northwest_territories"
        },
        {
          label: "Nova Scotia", value: "nova_scotia"
        },
        {
          label: "Ontario", value: "ontario"
        },
        {
          label: "Quebec", value: "quebec"
        }
      ]),

      // Quebec
      // Brunswick
      // Labrador
      // Edward Island
      params: {
        province: "british_columbia", low: 115.0,
        high: 0.0,
        begin_date: "2023-04-05", end_date: "2023-04-06"
      },
      priceLabelLeft: computed(() => ` ${range.value.min} $`),
      priceLabelRight: computed(() => `${range.value.max} $`)
    };
  },
  methods: {
    arrowButton() {
      const inputDate = this.date;
      const parts = inputDate.split("/");
      const year = parts[0];
      const month = String(parts[1]).padStart(2, "0");
      const day = String(parts[2]).padStart(2, "0");
      const formattedStartDate = `${year}-${month}-${day}`;

      const inputEndDate = this.date2;
      const parts2 = inputEndDate.split("/");
      const year2 = parts2[0];
      const month2 = String(parts2[1]).padStart(2, "0");
      const day2 = String(parts2[2]).padStart(2, "0");
      const formattedEndDate = `${year2}-${month2}-${day2}`;

      // setting the params json for axios request.
      this.params.province = this.options.value;
      this.params.low = this.range.min;
      this.params.high = this.range.max;
      this.params.begin_date = formattedStartDate;
      this.params.end_date = formattedEndDate;

      // Store the values in Vuex.
      this.store.dispatch("planner/updateFormattedDate", formattedStartDate);
      this.store.dispatch("planner/updateBudget", Number(this.budget));
      this.store.dispatch("planner/updateModelInit", this.params);

      let hotels_init = {};
      hotels_init.province = this.params.province;

      this.store.dispatch("planner/updateHotelInit", hotels_init);
    }
  }
});
