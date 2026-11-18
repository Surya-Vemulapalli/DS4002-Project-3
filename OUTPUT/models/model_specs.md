# Model specifications

OLS (statsmodels formula API), HC3 heteroskedasticity-robust standard errors,
fit on the training rows of `DATA/final/analysis_data.csv` only (`split == "train"`).

Baselines: room type = Entire home/apt; borough = Manhattan (`bronx_si` = Bronx + Staten Island combined). Photo scores are standardized with the training-set mean and SD.

## M1 baseline (`m1.pickle`)

```
log_price ~ private_room + bedrooms + bathrooms + shared_bath + accommodates + brooklyn + queens + bronx_si + min30 + private_room:bedrooms
```

- Number of parameters (incl. intercept): 11
- Training rows used: 1578
- Terms: `Intercept`, `private_room`, `bedrooms`, `bathrooms`, `shared_bath`, `accommodates`, `brooklyn`, `queens`, `bronx_si`, `min30`, `private_room:bedrooms`

## M2 photo model (`m2.pickle`)

```
log_price ~ private_room + bedrooms + bathrooms + shared_bath + accommodates + brooklyn + queens + bronx_si + min30 + private_room:bedrooms + z_musiq_koniq + z_musiq_ava + z_clipiqa
```

- Number of parameters (incl. intercept): 14
- Training rows used: 1578
- Terms: `Intercept`, `private_room`, `bedrooms`, `bathrooms`, `shared_bath`, `accommodates`, `brooklyn`, `queens`, `bronx_si`, `min30`, `private_room:bedrooms`, `z_musiq_koniq`, `z_musiq_ava`, `z_clipiqa`

## M3 interaction model (`m3.pickle`)

```
log_price ~ private_room + bedrooms + bathrooms + shared_bath + accommodates + brooklyn + queens + bronx_si + min30 + private_room:bedrooms + z_musiq_koniq + z_musiq_ava + z_clipiqa + z_musiq_koniq:brooklyn + z_musiq_koniq:queens + z_musiq_koniq:bronx_si + z_musiq_ava:brooklyn + z_musiq_ava:queens + z_musiq_ava:bronx_si + z_clipiqa:brooklyn + z_clipiqa:queens + z_clipiqa:bronx_si
```

- Number of parameters (incl. intercept): 23
- Training rows used: 1578
- Terms: `Intercept`, `private_room`, `bedrooms`, `bathrooms`, `shared_bath`, `accommodates`, `brooklyn`, `queens`, `bronx_si`, `min30`, `private_room:bedrooms`, `z_musiq_koniq`, `z_musiq_ava`, `z_clipiqa`, `z_musiq_koniq:brooklyn`, `z_musiq_koniq:queens`, `z_musiq_koniq:bronx_si`, `z_musiq_ava:brooklyn`, `z_musiq_ava:queens`, `z_musiq_ava:bronx_si`, `z_clipiqa:brooklyn`, `z_clipiqa:queens`, `z_clipiqa:bronx_si`
