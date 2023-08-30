import axios from 'axios';
import { defineComponent } from "vue";
import '@fortawesome/fontawesome-free/css/all.css'
import '@fortawesome/fontawesome-free/js/all.js'

export default defineComponent({
  name: "WeatherCard",
  data() {
    return {
      location: "",
      temp: 0,
      condition: "",
      feelsLike: 0,
      humidity: 0,
      wind: 0,
      datetime: "",
    };
  },
  props:["lat", "long"],
  mounted() {
    this.getWeatherData();
  },
  methods: {
    getWeatherData() {
      const options = {
        method: 'GET',
        url: 'https://weatherbit-v1-mashape.p.rapidapi.com/current',
        params: {
          lat: this.lat,
          lon: this.long
        },
        headers: {
          'content-type': 'application/octet-stream',
          'X-RapidAPI-Key': '9ed674538amshdcb9838f5a86f98p11ff43jsnbbfd1b11af6d',
          'X-RapidAPI-Host': 'weatherbit-v1-mashape.p.rapidapi.com'
        }
      };

      axios
        .request(options)
        .then((response) => {
          const data = response.data;
          const currentConditions = data.data[0];
          this.location = `${currentConditions.city_name}`;
          this.datetime = currentConditions.observation_time;
          this.temp = currentConditions.temp;
          this.condition = currentConditions.weather.description;
          this.feelsLike = currentConditions.app_temp || currentConditions.temp;
          this.humidity = currentConditions.rh;
          this.wind = currentConditions.wind_spd;
          console.log(data)
        })
        .catch((error) => {
          console.error(error);
        });
    },
  },
});
