# Smart Estate & Property Value Predictor

Building AI course project

## Summary

This project uses machine learning and neural networks to predict real estate property prices accurately based on key features like area, sauna size, distance to water, bathrooms, and neighborhood proximity, helping buyers and sellers make fair, data-driven decisions.

## Background

Estimating the fair market value of a property or cabin is often difficult for everyday buyers and sellers. Prices are usually guessed subjectively or rely on expensive real estate appraisals.

Key problems this project solves:
* **Unfair Pricing:** Helps buyers avoid overpaying and helps sellers set competitive, realistic prices.
* **Lack of Transparency:** Shows clearly how specific features (such as distance to water or number of bathrooms) impact property value.
* **Slow Valuation:** Provides instant price estimates instead of waiting days for manual assessments.

My motivation comes from learning linear regression, nearest neighbor algorithms, and neural networks in the Building AI course and applying them to solve a practical, everyday financial problem.

## How is it used?

Users (homebuyers, property owners, or real estate agents) enter basic details of a property into the model:
1. Total floor area ($m^2$)
2. Sauna or extra utility area ($m^2$)
3. Distance to the nearest water body ($m$)
4. Number of indoor bathrooms
5. Distance to the nearest neighbors ($m$)

The model processes these inputs and immediately outputs an estimated market price in euros/currency.

```python
import numpy as np

# Example property features: [size, sauna_size, distance_to_water, bathrooms, neighbor_distance]
property_features = np.array([82, 2, 65, 3, 516])

# Learned model coefficients (example from linear regression)
coefficients = np.array([3000, 200, -50, 5000, 100])

predicted_price = property_features @ coefficients
print(f"Estimated Property Price: {predicted_price}")
