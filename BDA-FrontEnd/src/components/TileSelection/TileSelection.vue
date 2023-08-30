<template>
  <div class="category-grid" style="margin: auto">
    <div class="card-grid" :class="chosenArray.length === 5 ? 'disabled' : ''">
      <div v-for="(cat, index) in category" :key="index" class="category-row-item-1">
        <q-card :class="'card-' + index"  class="card">
          <div class="q-card-category-column">
            <div class="q-card-category-row-1 text-weight-bold">
              <p style="text-align: center; padding-top: 10px">
                {{ cat.display_name }}
              </p>
            </div>
            <q-separator dark inset />

            <div class="q-card-category-row-2">
              <q-card-actions align="center">
                <q-btn flat :disable="cat.disabled" class="btn-text q-btn-custom" @click="AddList(cat.id, index)">Select</q-btn>
                <q-btn flat :disable="!cat.disabled" class="btn-text q-btn-custom" @click="removeList(cat.id, index)">Remove
                </q-btn>
              </q-card-actions>
            </div>
          </div>
        </q-card>
      </div>
    </div>
    <q-separator vertical></q-separator>
    <div class="category-row-item-2">
      <p  style="font-size: large; text-align: center; font-weight: bold">SELECTED CATEGORY</p>
      <transition-group name="fade" tag="div" mode="out-in">
        <div  class="text-weight-bold item2" v-for="(item, index) in chosenArray" :key="index">
          <div :class="!sliderStatus ? 'card-css' : ''">
            {{ item.display_name }}
          </div>
          <q-btn style="padding-right: 8px;" icon="highlight_off" flat square @click="removeChosen(item.display_name,index)"></q-btn>
          <q-slider
            v-if="sliderStatus"
            class="q-mt-xl"
            v-model="priceModel[index]"
            color="#1560bd"
            :marker-labels="arrayMarkerLabel"
            :min="1"
            :max="5"
          />
        </div>
      </transition-group>

    </div>
  </div>
</template>

<script type="text/javascript" src="./TileSelection.js"></script>
<style scoped lang="scss" src="./TileSelection.scss"></style>
