<template>
  <div class="search-style">
    <div class="row">
      <div class="col">
        <div class="q-pa-md" style="max-width: 600px">
          <q-input outlined v-model="budget" label="Total Budget $"></q-input>
        </div>
      </div>
      <div class="col">
        <div class="q-pa-md" style="max-width: 600px">
          <div class="q-gutter-md">
            <q-select outlined v-model="options.value" :options="options" label="Province" emit-value map-options />
          </div>
        </div>
      </div>
    </div>
    <div class="row">
      <div class="col">
        <div class="q-pa-md" style="max-width: 600px">
          <q-input outlined v-model="date" mask="date" :rules="['date']" label="Start-Date">
            <template v-slot:append>
              <q-icon name="event" class="cursor-pointer">
                <q-popup-proxy cover transition-show="scale" transition-hide="scale">
                  <q-date v-model="date">
                    <div class="row items-center justify-end">
                      <q-btn v-close-popup label="Close" color="primary" flat />
                    </div>
                  </q-date>
                </q-popup-proxy>
              </q-icon>
            </template>
          </q-input>
        </div>
      </div>
      <div class="col">
        <div class="q-pa-md" style="max-width: 600px">
          <q-input outlined v-model="date2" mask="date" :rules="['date']" label="End-Date">
            <template v-slot:append>
              <q-icon name="event" class="cursor-pointer">
                <q-popup-proxy cover transition-show="scale" transition-hide="scale">
                  <q-date v-model="date2">
                    <div class="row items-center justify-end">
                      <q-btn v-close-popup label="Close" color="primary" flat />
                    </div>
                  </q-date>
                </q-popup-proxy>
              </q-icon>
            </template>
          </q-input>
        </div>
      </div>
    </div>

    <div class="slider-style">
      <q-list dense>
        <q-item>
          <q-item-section avatar>
            <q-icon color="" name="attach_money"></q-icon>
          </q-item-section>
          <q-item-section>
            <q-range
              v-model="range"
              :min="0"
              :max="1000"
              label-always
              :left-label-value="priceLabelLeft"
              :right-label-value="priceLabelRight"
              drag-range
            ></q-range>
          </q-item-section>
        </q-item>

      </q-list>
    </div>


  </div>
</template>

<script type="text/javascript" src="./SearchBar.js"></script>
<style scoped lang="scss" src="./SearchBar.scss"></style>
