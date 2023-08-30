<template>
  <div :class="cssToggle ?'card-container-grid-1': 'card-container-grid'">

    <div class="card-container-row row-container-1">

      <div class="card-item row-1-item-1">
        <div class="q-pa-md">
          <div class="q-gutter-y-md column">
            <q-rating
              v-model="ratingModel"
              size="1.5em"
              color="orange"
              icon="star_border"
              icon-selected="star"
              icon-half="star_half"
              readonly
            />
          </div>
        </div>
      </div>

      <div class="card-item row-1-item-2">
        <div v-if="isDayActivity === 'Morning'">
          <img src="../../assets/c1-sunrise.gif" class="img-dim">
          <q-tooltip anchor="top middle" self="bottom middle" class="bg-blue text-body2">
             Activities preferred from 8:00 AM to 2:00 PM
          </q-tooltip>
        </div>
        <div v-else>
          <img src="../../assets/c3-even.gif" class="img-dim">
          <q-tooltip anchor="top middle" self="bottom middle" class="bg-blue text-body2">
            Activities preferred from 2:00 PM to 11:00 PM
          </q-tooltip>
        </div>

      </div>

    </div>

    <div class="card-container-row row-container-2">

      <div class="card-item row-2-item-1">
        <img :src="imageSrc" class="main-img">
      </div>

      <div class="card-item row-2-item-2">
        <p class="date-text">{{ dateText }}</p>
      </div>
    </div>

    <div class="card-container-row row-container-3">
      <p class="place-name truncate row-3-item-1" style="padding: 10px; font-size: 1em; ">{{ placeText }}</p>
      <q-tooltip anchor="top middle" self="bottom middle" class="bg-blue text-body2">
        {{ placeText }}
      </q-tooltip>
    </div>

    <div class="card-container-row row-container-4">
      <p class="time-text truncate row-4-item-1" style="padding: 0px 10px; font-size: 0.9em">{{ timeText }}</p>
      <q-tooltip anchor="top middle" self="bottom middle" class="bg-blue text-body2">
        {{ timeText }}
      </q-tooltip>
      <p class="cost-text row-4-item-2" style="font-size: 0.9em">{{ costText }}$</p>
    </div>


    <div v-if="enableLoc" class="card-container-row row-container-5">
      <q-btn class="round-button ripple" @click="fixed = true" rounded label="Peak At Weather" />
      <q-btn v-if="locToggle" @click="addToLocation()" class="round-button ripple" style="background-color: #953553"
             rounded label="Get Time, Distance" />
      <q-btn v-if="calToggle" class="round-button ripple" @click="calendar = true" rounded color="primary"
             label="Add to Calendar" />
    </div>

    <div v-if="disableLoc" class="card-container-row row-container-5">
      <q-btn v-if="removeToggle" @click="removeFromLocation()" class="round-button ripple" rounded color="red"
             label="Remove Location" />
    </div>

    <div v-if="weatherLoc" class="card-container-row row-container-5">
      <q-btn class="round-button ripple" @click="fixed = true" rounded label="Peak At Weather" />
    </div>
  </div>

  <q-dialog v-model="fixed" transition-show="fade" transition-hide="rotate">
    <div class="weather-container">
      <WeatherCard :lat="lat" :long="long" class="weather-actual"></WeatherCard>
      <q-btn class="weather-ok" label="OK" color="primary" v-close-popup />
    </div>
  </q-dialog>


  <q-dialog v-model="calendar" transition-show="fade" transition-hide="rotate">
    <div class="cal-container">
      <p>DATE : {{ dateText }}</p>
      <q-input outlined rounded v-model="start_time" mask="time" label="Start Time" :rules="['time']">
        <template v-slot:append>
          <q-icon name="access_time" class="cursor-pointer">
            <q-popup-proxy cover transition-show="scale" transition-hide="scale">
              <q-time v-model="start_time" landscape>
                <div class="row items-center justify-end">
                  <q-btn v-close-popup label="Close" color="primary" flat />
                </div>
              </q-time>
            </q-popup-proxy>
          </q-icon>
        </template>
      </q-input>

      <q-input outlined rounded v-model="end_time" mask="time" label="End Time" :rules="['time']">
        <template v-slot:append>
          <q-icon name="access_time" class="cursor-pointer">
            <q-popup-proxy cover transition-show="scale" transition-hide="scale">
              <q-time v-model="end_time" landscape>
                <div class="row items-center justify-end">
                  <q-btn v-close-popup label="Close" color="primary" flat />
                </div>
              </q-time>
            </q-popup-proxy>
          </q-icon>
        </template>
      </q-input>
      <q-btn @click="addEventInCal()" class="weather-ok" label="Add to Calendar" color="primary" v-close-popup />
    </div>
  </q-dialog>
</template>

<script type="text/javascript" src="./ResultCard.js"></script>
<style scoped lang="scss" src="./ResultCard.scss"></style>
