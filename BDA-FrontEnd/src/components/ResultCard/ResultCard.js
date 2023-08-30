import { defineComponent, onMounted, ref } from "vue";
import WeatherCard from "components/Weather/WeatherAPI.vue";
import { useStore } from "vuex";

/*
* todo : Dynamic Image case for main image
*        Format for date
*        truncate logic
* */

export default defineComponent({
  name: "ResultCard",
  components: { WeatherCard },
  props: [
    "ratingModel",
    "imageSrc",
    "isDayActivity",
    "dateText",
    "placeText",
    "timeText",
    "costText",
    "lat",
    "long",
    "enableLoc",
    "disableLoc",
    "weatherLoc",
    "calToggle",
    "locToggle",
    "cssToggle",
    "removeToggle"
  ],
  setup() {
    const store = useStore();
    return {
      store,
      fixed: ref(false),
      calendar: ref(false),
      end_time: ref("10:56"),
      start_time: ref("10:56")
    };

  },
  methods: {
    findKey(title) {
      const distLen = this.store.getters["planner/getCatCard"];
      let finKey = 0;
      // Loop through each key in the cat object
      for (const key in distLen) {
        if (distLen.hasOwnProperty(key)) {
          const catArray = distLen[key];
          catArray.forEach((obj) => {
            if (obj.name == title) {
              finKey = key;
            }
          });
        }
      }
      return finKey;
    },

    setDistDetails(origin, dest, keyDist, title1, title2, org, des) {

      const service = new google.maps.DistanceMatrixService();
      const distanceMatrixPromise = new Promise((resolve, reject) => {
        service.getDistanceMatrix(
          {
            origins: origin,
            destinations: dest,
            travelMode: 'DRIVING',
          }, (response, status) => {
            if (status === 'OK') {
              resolve(response);
            } else {
              reject(new Error(`Distance Matrix request failed: ${status}`));
            }
          });
      });

      distanceMatrixPromise.then((response) => {
        // Process the response
          let origins = response.originAddresses;
          let destinations = response.destinationAddresses;

          for (let i = 0; i < origins.length; i++) {
            let results = response.rows[i].elements;
            for (let j = 0; j < results.length; j++) {
              let element = results[j];
              let distance = element.distance.text;
              let duration = element.duration.text;
              let from = origins[i];
              let to = destinations[j];
              console.log(distance);
              console.log(duration);
              console.log(from);
              console.log(to);
              let demo = {
                "origin": org,
                "dest": des,
                "key": keyDist,
                "distance": distance,
                "duration": duration,
                "from": from,
                "to": to,
                "title1": title1,
                "title2": title2,
              }
              this.$store.dispatch("planner/updateAddMapCard", demo);
            }
          }
      }).catch((error) => {
        console.error(error);
      });
    },

    updateLocationDS() {
      const distLen = this.store.getters["planner/getDistCard"];
      let keyDist = this.findKey(this.placeText);
      console.log(this.placeText);
      console.log(keyDist);
      if (distLen[keyDist].length == 2) {
        let org = [distLen[keyDist][0].location[0], distLen[keyDist][0].location[1]]
        let des = [distLen[keyDist][1].location[0], distLen[keyDist][1].location[1]]
        let origin1 = new google.maps.LatLng(distLen[keyDist][0].location[0], distLen[keyDist][0].location[1])
        let dest1 = new google.maps.LatLng(distLen[keyDist][1].location[0], distLen[keyDist][1].location[1])
        let origin = [origin1];
        let dest = [dest1];
        let title1 = distLen[keyDist][0].name;
        let title2 = distLen[keyDist][1].name;
        this.setDistDetails(origin, dest, keyDist, title1, title2, org, des);
      }
      if (distLen[keyDist].length == 3) {
        let org = [distLen[keyDist][1].location[0], distLen[keyDist][1].location[1]]
        let des = [distLen[keyDist][2].location[0], distLen[keyDist][2].location[1]]
        let origin1 = new google.maps.LatLng(distLen[keyDist][1].location[0], distLen[keyDist][1].location[1]);
        let dest1 = new google.maps.LatLng(distLen[keyDist][2].location[0], distLen[keyDist][2].location[1])
        let origin = [origin1];
        let dest = [dest1];
        let title1 = distLen[keyDist][1].name;
        let title2 = distLen[keyDist][2].name;
        this.setDistDetails(origin, dest, keyDist, title1, title2, org, des);
      }
      if (distLen[keyDist].length == 4) {
        let org = [distLen[keyDist][2].location[0], distLen[keyDist][2].location[1]]
        let des = [distLen[keyDist][3].location[0], distLen[keyDist][3].location[1]]
        let origin1 = new google.maps.LatLng(distLen[keyDist][2].location[0], distLen[keyDist][2].location[1])
        let dest1 = new google.maps.LatLng(distLen[keyDist][3].location[0], distLen[keyDist][3].location[1])
        let origin = [origin1];
        let dest = [dest1];
        let title1 = distLen[keyDist][2].name;
        let title2 = distLen[keyDist][3].name;
        this.setDistDetails(origin, dest, keyDist, title1, title2, org, des);
      }
    },
    addToLocation() {
      this.store.dispatch("planner/updateAddLocation", this.placeText);
      this.updateLocationDS();
    },
    removeFromLocation() {
      this.store.dispatch("planner/updateToggleLoc", this.placeText);
      this.store.dispatch("planner/updateRemoveLocation", this.placeText);
      let keyDist = this.findKey(this.placeText);
      this.store.dispatch("planner/updateRemoveMapCard", keyDist);
    },
    addEventInCal() {
      const startDate = `${this.dateText}T${this.start_time}`;
      const endDate = `${this.dateText}T${this.end_time}`;

      // trigger disable event

      this.store.dispatch("planner/updateToggleCal", this.placeText);

      const event = {
        "title": this.placeText,
        "start": new Date(startDate),
        "end": new Date(endDate),
        "class": "sport",
        "content": "Budget: " + this.costText + "$"
      };
      this.$emit("addToEvent", event, this.costText);
    }
  }
});




