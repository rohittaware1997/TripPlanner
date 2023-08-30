import { defineComponent, ref } from "vue";
import { useStore } from "vuex";


export default defineComponent({
  name: "GoogleMap",
  props: ["keyIndex"],
  setup() {
    const store = useStore();
    const mapDet= ref();
    return {
      store,
      mapDet
    };
  },
  data() {
    return {
      map: null,
      markers: [],
      directions: null
    };
  },
  mounted() {
    const demo = this.store.getters["planner/getMapCard"];
    this.mapDet = demo[this.keyIndex];
    this.initMap();
  },
  methods: {
    initMap() {
      // Create the map
      let mapCard = this.store.getters["planner/getMapCard"];
      console.log(mapCard[this.keyIndex]);

      this.map = new google.maps.Map(this.$refs.map, {
        center: { lat: mapCard[this.keyIndex][0].origin[0], lng: mapCard[this.keyIndex][0].origin[1] },
        zoom: 7
      });

      // Add the markers
      const points = [];

      const demoMapCard = mapCard[this.keyIndex];
      demoMapCard.forEach((obj) => {
          let pt = {
            "lat": obj.origin[0],
            "lng": obj.origin[1],
          }
        let pt1 = {
          "lat": obj.dest[0],
          "lng": obj.dest[1],
        }
          points.push(pt);
          points.push(pt1);
      });
      console.log(points);
      for (let i = 0; i < points.length; i++) {
        const marker = new google.maps.Marker({
          position: points[i],
          map: this.map
        });
        this.markers.push(marker);
      }
    }
  }
});
