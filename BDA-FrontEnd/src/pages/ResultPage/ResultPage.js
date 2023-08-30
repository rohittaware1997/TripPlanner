import { defineComponent, onMounted, ref } from "vue";
import ResultCard from "components/ResultCard/ResultCard.vue";
import draggable from "vuedraggable";
import PlannerCalendar from "components/PlannerCalendar/PlannerCalendar.vue";
import { useStore } from "vuex";
import ParallaxScroll from "components/ParallaxScroll/ParallaxScroll.vue";
import jsPDF from "jspdf";
import "jspdf-autotable";
import html2canvas from "html2canvas";
import GoogleMap from "components/GoogleMap/GoogleMap.vue";


export default defineComponent({
  name: "ResultPage",
  components: { ResultCard, draggable, PlannerCalendar, ParallaxScroll, GoogleMap },
  mounted() {
    const clickedDiv = document.querySelector(".vuecal__menu");
    clickedDiv.style.backgroundColor = "#58b4a9";
  },
  updated() {
    const clickedDiv = document.querySelectorAll(".vuecal__event");
    if (clickedDiv !== null) {
      for (let i = 0; i < clickedDiv.length; i++) {
        clickedDiv[i].style.backgroundColor = "aquamarine";
      }
    }
  },
  data() {
    return {
      events: [],
      width: 100,
      bgColor: "#58b4a9"
    };
  },
  setup() {
    const cat = ref();
    const rec = ref();
    const dict = ref();
    const map = ref();
    const store = useStore();
    const printClass = ref(false);
    let globalBudget = ref("");
    const startDate = ref("");
    const visible = ref(false);

    onMounted(() => {
      store.dispatch("planner/updateMapDistCard");
      globalBudget.value = store.getters["planner/getBudget"];
      cat.value = store.getters["planner/getCatCard"];
      rec.value = store.getters["planner/getRecCard"];
      startDate.value = store.getters["planner/getStartDate"];
      dict.value = store.getters["planner/getDistCard"];
      map.value = store.getters["planner/getMapCard"];
    });

    return {
      cat,
      rec,
      store,
      dict,
      map,
      startDate,
      globalBudget,
      printClass,
      slide: ref(1),
      tab: ref("activities"),
      lowest: ref(false),
      home: ref(false),
      save: ref(false),
      maps: ref(false),
      maximizedToggle: ref(true),
      visible,
      keyIndex: ref(0)
    };
  },
  methods: {
    toggleMaps(index){
      this.keyIndex = index
      this.maps = true;
    },
    finalAddEvent(event, budget) {
      console.log(event);
      this.events.push(event);
      this.store.dispatch("planner/removeFromBudget", budget);
      this.globalBudget = this.store.getters["planner/getBudget"];
      if (this.globalBudget > 0) {
        this.width -= budget / 10;
        if (this.width >= 80) {
          this.bgColor = "#58b4a9";
        } else if (this.width >= 30 && this.width < 80) {
          this.bgColor = "#fdb813";
        } else if (this.width < 26) {
          this.bgColor = "#FF5E0E";
        }
      } else {
        alert("You are above the budget, you can still continue");
        this.width = 100;
        this.bgColor = "#FF0000";
      }
    },
    homePage() {
      // clean up code
      this.store.dispatch("planner/updateAllValues");
      // toggle first page
      this.$emit("backFirstPage", true);
    },
    async saveItn() {
      // Create a new jsPDF instance
      this.printClass = true;
      this.visible = true;
      setTimeout(async () => {
        const content = document.querySelector(".print-container");
        const canvas = await html2canvas(content);
        const imgData = canvas.toDataURL("image/png");
        const pdf = new jsPDF();
        pdf.addImage(imgData, "PNG", 0, 0, 210, 297);
        pdf.save("itinerary.pdf");
        this.printClass = false;
        this.visible = false;
      }, 200);
    },
    removeEventCal(title, budget) {
      const index = this.events.findIndex(item => item.title == title);
      this.events.splice(index, 1);
      console.log(budget);
      this.store.dispatch("planner/addFromBudget", budget);
      this.globalBudget = this.store.getters["planner/getBudget"];
      let letActWidth = 0;
      if (this.globalBudget > 0) {
        if (this.width == 100) {
          this.width = 0;
        }
        this.width += budget / 10;
        if (this.width < 26) {
          this.bgColor = "#FF5E0E";
        } else if (this.width >= 30 && this.width < 70) {
          this.bgColor = "#fdb813";
        } else if (this.width >= 70) {
          this.bgColor = "#58b4a9";
        }
      }
    }
  }
});
