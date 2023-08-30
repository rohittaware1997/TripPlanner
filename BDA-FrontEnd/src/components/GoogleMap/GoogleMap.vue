<template>
  <div style="text-align: center;">
    <p style="font-size: 3em; font-weight: bold;  text-stroke: 1px black;   text-shadow: 2px 2px 2px rgba(0, 0, 0, 0.5);" >Google Map</p>
  </div>
  <div class="map-container">
    <div ref="map" class="map"></div>
  </div>
  <div>
    <div>
      <p style="text-align: center; font-weight: bold; font-size: 3em; text-stroke: 1px black;   text-shadow: 2px 2px 2px rgba(0, 0, 0, 0.5);">Details</p>
      <q-separator style="background-color: white" ></q-separator>
      <ol style="font-size: 1.5em;" v-for="(details, index) in mapDet" :key="index">
        <ul>
          <li>
            Start: {{ details.from }}
          </li>
          <li>
            Destination: {{ details.to }}
          </li>
          <li>
            Distance: {{ details.distance }}
          </li>
          <li>
            Time: {{ details.duration }}
          </li>
        </ul>
        <q-separator style="background-color: white; width: 90%; margin-top: 20px" ></q-separator>
      </ol>
      <q-separator style="background-color: white" ></q-separator>
    </div>
  </div>
</template>



<script type="text/javascript" src="./GoogleMap.js"></script>
<style scoped lang="scss" src="./GoogleMap.scss"></style>
