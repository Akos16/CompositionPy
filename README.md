# Adatkezelő modulok

## 1. `xMaxDataHandler.py`

Ez a modul felelős az `xMaxData/` mappában található kísérleti adatok beolvasásáért. A fájlok elnevezése az energia-bin számozását követi a fájlnév végén (például: `XMaxDist_Ebin0.txt`).

### Új adatfájl hozzáadása

Amennyiben több $X_{\max}$ adatsor áll rendelkezésre, azokat ugyanezzel az elnevezési konvencióval kell a mappába helyezni, majd a kód `match i` szerkezetét ki kell egészíteni az új indexszel.

### Főbb függvények

#### `getXmaxData()`

Beolvassa és strukturált formában adja vissza a kísérleti adatokat.

#### `moments_from_prob()`

Kiszámítja az eloszlás következő momentumait:

- **átlag** (`mean`)
- **második centrális momentum** (`mu2`)
- **ferdeség** (`skewness`)
- **excess kurtosis**

#### `moments_with_errors()`

Ugyanezeket a momentumokat számolja ki a hozzájuk tartozó hibaértékekkel együtt, **5000 toy Monte Carlo** szimuláció segítségével.

### Használati példa

```python
xMaxData = xMaxHandler.getXmaxData()

# Struktúra:
# xMaxData[fájl_index][oszlop_index][:48]

# Oszlopok:
# 0: Xmax
# 1: Counts
# 2: CountsSqrt

minta_adat = xMaxData[0][1][:48] #0-ik fájl, Counts oszlop
```

---

## 2. `monteCarloDataHandler.py`

Ez a modul kezeli a szimulált Monte Carlo adatfájlokat.

### Főbb függvények

#### `readAndRebin()`

Újrabinezi a szimulációs eloszlásokat

$$
504~\mathrm{g/cm^2}
$$

értéktől kezdődően, $12~\mathrm{g/cm^2}$ lépésközzel.

#### `getMonteCarloData()`

A `getXmaxData()`-hoz hasonlóan `match i` logika alapján olvassa be az MC fájlokat (például `component0`).

### Visszatérési érték

A visszatérési érték első **4 eleme** a különböző MC komponensfájloknak felel meg, míg az **5. elem** az $X_{\max}$ tengelyt tartalmazza

$$
500~\mathrm{g/cm^2} \leq X_{\max} \leq 1500~\mathrm{g/cm^2}.
$$

### Használati példa

```python
monteCarloData = monteCarloDataHandler().getMonteCarloData()

# Struktúra:
# monteCarloData[MC_fájl_index][lgE_index][:48]

arr1 = monteCarloData[0][1][:48]
# Első MC fájl, lgE = 18.0-s energiabin
```

---

## 3. `Main.py` integráció

A főprogram indításakor mindkét adatkezelő osztály inicializálásra kerül.

Mivel **3 Pierre Auger $X_{\max}$ adatfájl** dolgozható fel, és a legkisebb elemzendő energia

$$
\lg(E/\mathrm{eV}) = 18.0,
$$

ezért a feldolgozó `for` ciklus az **1-es indexű energiabintől** indul, és ezen keresztül lépked végig az adatokon. # monteCarloData[MC_fájl_index][lgE_index][:48] -> itt lgEindex 1

### Ciklusbeli lekérdezések a megfelelő energia-binekre

```python
# lgE = 18.0 (index 1)
arr_lgE_18_0 = monteCarloData[0][1][:48]

# lgE = 18.5 (index 2)
arr_lgE_18_5 = monteCarloData[0][2][:48]
```

### Adatszerkezetek összefoglalása

#### Kísérleti adatok

```text
xMaxData[fájl_index][oszlop_index][:48]
```

| Oszlop index | Tartalom |
|--------------|----------|
| `0` | `Xmax` |
| `1` | `Counts` |
| `2` | `CountsSqrt` |

#### Monte Carlo adatok

```text
monteCarloData[MC_fájl_index][lgE_index][:48]
```

ahol:

- `MC_fájl_index` az adott Monte Carlo komponenst azonosítja,
- `lgE_index` az energia-bin indexe,
- `[:48]` az első 48 releváns bin kiválasztását jelenti.
