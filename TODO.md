# Tasks

## In Progress

### Energy Load Model
- [ ] Prepare an initial dataset for training, based on calendar features and temperature

---

## Future Plans
- [ ] Full development cycle for an onshore wind model
- [ ] Full development cycle for a solar model
- [ ] Full development cycle for a price model
- [ ] Update and expand project documentation (README.md, architecture.md, etc.)

---

## Completed

### General
- [x] Set up repository structure
- [x] Implement data collection function from ENTSO-E
- [x] Implement data collection function for installed capacities
- [x] Implement functions to collect weather data using Open-Meteo Historical Forecast

### Offshore Wind Model
- [x] Initial data collection and feature engineering
- [x] Evaluate features for signal strength and establish model baselines
- [x] Experimental selection of the feature space
- [x] Experimental selection of the sliding window configuration
- [x] ALTERNATIVE: Test the hypothesis that interpolating data points is superior to duplicating them with unique timestamps
- [x] Hyperparameter tuning and testing of custom loss functions
- [x] Model validation on a holdout set; finalize operational modes and hyperparameters
- [x] Deploy to production

### Web Interface
- [x] Create an MVP of a publicly accessible, daily-updated interface