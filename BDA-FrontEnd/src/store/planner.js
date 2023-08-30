import "../store/constData.js";

const state = {
  budget: 0,
  recCard: {},
  catCard: {},
  hotelCard: [],
  distCard: {},
  mapCard: {},
  model_parameter: [],
  model_init_param: {},
  hotel_parameter: [],
  hotel_init_param: {},
  formattedStartDate: ""
};
const mutations = {
  setAddMapCard(state, data) {
    state.mapCard[data.key].push(data);
  },
  setRemoveMapCard(state, key) {
      state.mapCard[key].pop();
  },
  setRemoveFromBudget(state, budget) {
    state.budget -= budget.toFixed(2);
    state.budget.toFixed(2);
  },
  setBudget(state, budget) {
    state.budget = budget;
    state.budget.toFixed(2);
  },
  setRecCard(state, recCard) {
    state.recCard = recCard;
  },
  setCatCard(state, catCard) {
    state.catCard = catCard;
  },
  setModelInit(state, model_init_param) {
    state.model_init_param = model_init_param;
  },
  setModelParam(state, model_parameter) {
    state.model_init_param.cat_rating = model_parameter;
    const finalJSON = JSON.stringify(state.model_init_param);
    state.model_parameter.push(finalJSON);
  },
  setHotelInit(state, hotel_init_param) {
    state.hotel_init_param = hotel_init_param;
  },
  setHotelParam(state, hotel_parameter) {
    state.hotel_init_param.amenities = hotel_parameter;
    const finalJSON = JSON.stringify(state.hotel_init_param);
    state.hotel_parameter.push(finalJSON);
  },
  resetValues(state) {
    state.budget = 0;
    state.recCard = {};
    state.catCard = {};
    state.hotelCard = [];
    state.mapCard = {};
    state.distCard = {};
    state.model_parameter = [];
    state.model_init_param = {};
    state.hotel_parameter = [];
    state.hotel_init_param = {};
    state.formattedStartDate = "";
  },
  setAddFromBudget(state, budget) {
    state.budget += budget;
  },
  setFormattedDate(state, date) {
    state.formattedStartDate = date;
  },
  setToggleCal(state, title) {
    // Loop through each key in the cat object
    for (const key in state.catCard) {
      // Check if the key is a valid property of the object
      if (state.catCard.hasOwnProperty(key)) {
        // Access the array of objects for the current key
        const catArray = state.catCard[key];
        // Loop through each object in the array
        catArray.forEach((obj) => {
          // Check if the object's title matches the titleToMatch variable
          if (obj.name === title) {
            // Update the toggle value to the new value
            obj.calendarToggle = !obj.calendarToggle;
          }
        });
      }
    }
  },
  setToggleLoc(state, title) {
    for (const key in state.catCard) {
      // Check if the key is a valid property of the object
      if (state.catCard.hasOwnProperty(key)) {
        // Access the array of objects for the current key
        const catArray = state.catCard[key];
        // Loop through each object in the array
        catArray.forEach((obj) => {
          // Check if the object's title matches the titleToMatch variable
          if (obj.name === title) {
            // Update the toggle value to the new value
            obj.locationToggle = !obj.locationToggle;
          }
        });
      }
    }
  },
  setMapCard(state, maps){
    state.mapCard = maps
  },
  setDistCard(state, dist){
    state.distCard = dist;
  },
  setMapDistCard(state) {
    for (const key in state.distCard) {
      // Check if the key is a valid property of the object
      if (state.distCard.hasOwnProperty(key)) {
        // Set the array value for the current key to an empty array
        state.distCard[key] = [];
      }
    }

    for (const key in state.mapCard) {
      // Check if the key is a valid property of the object
      if (state.mapCard.hasOwnProperty(key)) {
        // Set the array value for the current key to an empty array
        state.mapCard[key] = [];
      }
    }
  },
  setAddLocation(state, title) {
    for (const key in state.catCard) {
      if (state.catCard.hasOwnProperty(key)) {
        const distArray = state.catCard[key];
        distArray.forEach((obj) => {
          // Check if the object's title matches the titleToMatch variable
          if (obj.name === title) {
            // Update the toggle value to the new value
            if (state.distCard[key].length < 4) {
              state.distCard[key].push(obj);
              obj.locationToggle = !obj.locationToggle;
            } else {
              alert("You cannot add more items!!");
            }
          }
        });
      }
    }
  },
  setRemoveLocation(state, title) {
    for (const key in state.distCard) {
      if (state.distCard.hasOwnProperty(key)) {
        const index = state.distCard[key].findIndex(item => item.name == title);
        state.distCard[key].splice(index, 1);
      }
    }
  }
};

const actions = {
  updateDistCard({ commit }, dist){
    commit("setDistCard", dist);
  },
  updateMapCard({ commit }, map){
    commit("setMapCard", map)
  },
  updateAddMapCard({ commit }, demo) {
    commit("setAddMapCard", demo);
  },
  updateRemoveMapCard({ commit }, key) {
    commit("setRemoveMapCard", key);
  },
  updateBudget({ commit }, budget) {
    commit("setBudget", budget);
  },
  removeFromBudget({ commit }, budget) {
    commit("setRemoveFromBudget", budget);
  },
  addFromBudget({ commit }, budget) {
    commit("setAddFromBudget", budget);
  },
  updateRecCard({ commit }, recCard) {
    commit("setRecCard", recCard);
  },
  updateCatCard({ commit }, catCard) {
    commit("setCatCard", catCard);
  },
  updateModelInit({ commit }, model_init_param) {
    commit("setModelInit", model_init_param);
  },
  updateModelParam({ commit }, model_parameter) {
    commit("setModelParam", model_parameter);
  },
  updateHotelInit({ commit }, hotel_init_param) {
    commit("setHotelInit", hotel_init_param);
  },
  updateHotelParam({ commit }, hotel_parameter) {
    commit("setHotelParam", hotel_parameter);
  },
  updateAllValues({ commit }) {
    commit("resetValues");
  },
  updateFormattedDate({ commit }, date) {
    commit("setFormattedDate", date);
  },
  updateToggleCal({ commit }, title) {
    commit("setToggleCal", title);
  },
  updateToggleLoc({ commit }, title) {
    commit("setToggleLoc", title);
  },
  updateMapDistCard({ commit }) {
    commit("setMapDistCard");
  },
  updateAddLocation({ commit }, title) {
    commit("setAddLocation", title);
  },
  updateRemoveLocation({ commit }, title) {
    commit("setRemoveLocation", title);
  }

};
const getters = {
  getBudget(state) {
    return state.budget.toFixed(2);
  },
  getRecCard(state) {
    return state.recCard;
  },
  getCatCard(state) {
    return state.catCard;
  },
  getModelParam(state) {
    return state.model_parameter;
  },
  getHotelParam(state) {
    return state.hotel_parameter;
  },
  getStartDate(state) {
    return state.formattedStartDate;
  },
  getDistCard(state) {
    return state.distCard;
  },
  getMapCard(state){
    return state.mapCard;
  }
};

export default {
  namespaced: true,
  getters,
  mutations,
  actions,
  state
};
