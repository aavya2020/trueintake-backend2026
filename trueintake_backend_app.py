from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

APP_TITLE = "TrueIntake AI Backend"
APP_VERSION = "0.2.0"
FDC_BASE_URL = "https://api.nal.usda.gov/fdc/v1"
DSLD_BASE_URL = "https://api.ods.od.nih.gov/dsld/v9"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_FDC_DATA_TYPES = ["Foundation", "Branded", "SR Legacy", "Survey (FNDDS)"]

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="Backend for estimating actual nutrient intake from supplements and food using real DSID-4 regression coefficients plus USDA FoodData Central.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NUTRIENT_SYNONYMS: Dict[str, Dict[str, Any]] = {
    "calcium": {
        "canonical": "Calcium",
        "unit": "mg",
        "aliases": ["calcium", "ca"],
        "fdc_names": ["Calcium, Ca"],
    },
    "iron": {
        "canonical": "Iron",
        "unit": "mg",
        "aliases": ["iron", "fe"],
        "fdc_names": ["Iron, Fe"],
    },
    "magnesium": {
        "canonical": "Magnesium",
        "unit": "mg",
        "aliases": ["magnesium", "mg"],
        "fdc_names": ["Magnesium, Mg"],
    },
    "zinc": {
        "canonical": "Zinc",
        "unit": "mg",
        "aliases": ["zinc", "zn"],
        "fdc_names": ["Zinc, Zn"],
    },
    "vitamin_c": {
        "canonical": "Vitamin C",
        "unit": "mg",
        "aliases": ["vitamin c", "ascorbic acid"],
        "fdc_names": ["Vitamin C, total ascorbic acid"],
    },
    "vitamin_d": {
        "canonical": "Vitamin D",
        "unit": "iu",
        "aliases": ["vitamin d", "vitamin d3", "cholecalciferol"],
        "fdc_names": ["Vitamin D (D2 + D3)", "Vitamin D"],
    },
    "vitamin_b12": {
        "canonical": "Vitamin B-12",
        "unit": "mcg",
        "aliases": ["vitamin b12", "b12", "cobalamin", "vitamin b-12"],
        "fdc_names": ["Vitamin B-12"],
    },
    "folic_acid": {
        "canonical": "Folic Acid",
        "unit": "mcg",
        "aliases": ["folic acid", "folate", "dfe", "dietary folate equivalents"],
        "fdc_names": ["Folic acid", "Folate, total", "Folate, DFE"],
    },
    "vitamin_a": {
        "canonical": "Vitamin A",
        "unit": "iu",
        "aliases": ["vitamin a", "vitamin a, iu", "retinol", "rae"],
        "fdc_names": ["Vitamin A, RAE", "Retinol"],
    },
    "vitamin_e": {
        "canonical": "Vitamin E",
        "unit": "iu",
        "aliases": ["vitamin e", "alpha tocopherol"],
        "fdc_names": ["Vitamin E (alpha-tocopherol)"],
    },
    "copper": {
        "canonical": "Copper",
        "unit": "mg",
        "aliases": ["copper", "cu"],
        "fdc_names": ["Copper, Cu"],
    },
    "iodine": {
        "canonical": "Iodine",
        "unit": "mcg",
        "aliases": ["iodine", "i"],
        "fdc_names": ["Iodine, I"],
    },
}

UNIT_FACTORS_TO_CANONICAL: Dict[str, Dict[str, float]] = {
    "mg": {"mg": 1.0, "mcg": 0.001, "g": 1000.0},
    "mcg": {"mcg": 1.0, "mg": 1000.0, "g": 1_000_000.0},
    "g": {"g": 1.0, "mg": 0.001, "mcg": 0.000001},
    "iu": {"iu": 1.0},
}


class PredictSupplementRequest(BaseModel):
    category: str
    nutrient: str
    label_claim: float = Field(..., ge=0)
    unit: str
    servings_per_day: float = Field(default=1.0, gt=0)


class SupplementItem(BaseModel):
    category: str
    nutrient: str
    label_claim: float = Field(..., ge=0)
    unit: str
    servings_per_day: float = Field(default=1.0, gt=0)


class FoodItem(BaseModel):
    fdc_id: int
    grams: float = Field(default=100.0, gt=0)


class CalculateTotalIntakeRequest(BaseModel):
    supplements: List[SupplementItem] = Field(default_factory=list)
    foods: List[FoodItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("-", " ").replace(",", "").split())


def normalize_unit_token(unit: str) -> str:
    raw = normalize_text(unit)
    raw = raw.replace(".", "")
    raw = raw.replace("microgram", "mcg")
    raw = raw.replace("micrograms", "mcg")
    raw = raw.replace("milligram", "mg")
    raw = raw.replace("milligrams", "mg")
    raw = raw.replace("gram", "g")
    raw = raw.replace("grams", "g")

    if raw in {"mcg dfe", "mcg dietary folate equivalents"}:
        return "mcg_dfe"
    if raw in {"iu", "i u", "iu(s)"}:
        return "iu"
    if raw in {"calories", "calorie", "kcal"}:
        return "kcal"

    return raw


@lru_cache(maxsize=1)
def get_settings() -> Dict[str, str]:
    return {
        "fdc_api_key": os.getenv("FDC_API_KEY", "").strip(),
        "dsid_model_csv": os.getenv("DSID_MODEL_CSV", "./data/dsid_models.csv").strip(),
        "dsld_api_key": os.getenv("DSLD_API_KEY", "").strip(),
    }


@lru_cache(maxsize=1)
def load_dsid_models() -> pd.DataFrame:
    csv_path = get_settings()["dsid_model_csv"]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"DSID model CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {
        "category_code",
        "nutrient",
        "unit",
        "min_label_claim",
        "max_label_claim",
        "pred_pct_intercept",
        "pred_pct_linear_coeff",
        "pred_pct_quadratic_coeff",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"DSID model CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["category_norm"] = df["category_code"].astype(str).map(normalize_text)
    df["nutrient_norm"] = df["nutrient"].astype(str).map(normalize_text)
    df["unit_norm"] = df["unit"].astype(str).map(normalize_unit_token)
    return df


def resolve_canonical_nutrient(input_name: str) -> Dict[str, Any]:
    raw = normalize_text(input_name)
    for data in NUTRIENT_SYNONYMS.values():
        aliases = [normalize_text(a) for a in data["aliases"] + [data["canonical"]]]
        if raw in aliases:
            return data
    return {
        "canonical": input_name.strip(),
        "unit": None,
        "aliases": [input_name.strip()],
        "fdc_names": [input_name.strip()],
    }


def convert_amount(value: float, from_unit: str, to_unit: str, nutrient_name: Optional[str] = None) -> float:
    from_unit_norm = normalize_unit_token(from_unit)
    to_unit_norm = normalize_unit_token(to_unit)
    nutrient_norm = normalize_text(nutrient_name or "")

    if from_unit_norm == to_unit_norm:
        return value

    if to_unit_norm in UNIT_FACTORS_TO_CANONICAL and from_unit_norm in UNIT_FACTORS_TO_CANONICAL[to_unit_norm]:
        return value * UNIT_FACTORS_TO_CANONICAL[to_unit_norm][from_unit_norm]

    if nutrient_norm in {"vitamin d", "vitamin d3", "cholecalciferol"}:
        if from_unit_norm == "mcg" and to_unit_norm == "iu":
            return value * 40.0
        if from_unit_norm == "iu" and to_unit_norm == "mcg":
            return value / 40.0

    if nutrient_norm in {"vitamin a", "retinol"}:
        if from_unit_norm == "mcg" and to_unit_norm == "iu":
            return value / 0.3
        if from_unit_norm == "iu" and to_unit_norm == "mcg":
            return value * 0.3

    if nutrient_norm in {"folic acid", "folate"}:
        if from_unit_norm == "mcg_dfe" and to_unit_norm == "mcg":
            return value * 0.6
        if from_unit_norm == "mcg" and to_unit_norm == "mcg_dfe":
            return value / 0.6

    raise ValueError(f"Cannot convert from {from_unit} to {to_unit} for nutrient {nutrient_name}")


def get_model_row(category: str, nutrient: str) -> pd.Series:
    models = load_dsid_models()
    category_norm = normalize_text(category)
    nutrient_info = resolve_canonical_nutrient(nutrient)
    nutrient_norm = normalize_text(nutrient_info["canonical"])

    match = models[
        (models["category_norm"] == category_norm)
        & (models["nutrient_norm"] == nutrient_norm)
    ]

    if match.empty:
        available = models[["category_code", "nutrient"]].drop_duplicates().head(25).to_dict("records")
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"No DSID model found for category='{category}' nutrient='{nutrient_info['canonical']}'",
                "examples": available,
            },
        )

    return match.iloc[0]


def predict_from_model(category: str, nutrient: str, label_claim: float, unit: str, servings_per_day: float) -> Dict[str, Any]:
    row = get_model_row(category, nutrient)
    model_unit = row["unit"]

    converted_label_claim = convert_amount(
        label_claim,
        unit,
        model_unit,
        nutrient_name=row["nutrient"],
    )
    min_claim = float(row["min_label_claim"])
    max_claim = float(row["max_label_claim"])

    if converted_label_claim < min_claim or converted_label_claim > max_claim:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Label claim {converted_label_claim:g} {model_unit} is outside the supported model range "
                f"({min_claim:g} to {max_claim:g} {model_unit}) for {row['category_code']} / {row['nutrient']}."
            ),
        )

    intercept = float(row["pred_pct_intercept"])
    linear = float(row["pred_pct_linear_coeff"])
    quadratic = float(row["pred_pct_quadratic_coeff"])

    percent_diff = intercept + linear * converted_label_claim + quadratic * (converted_label_claim ** 2)
    predicted_per_serving = converted_label_claim * (1 + percent_diff / 100.0)
    predicted_per_day = predicted_per_serving * servings_per_day
    label_claim_per_day = converted_label_claim * servings_per_day

    return {
        "category": row["category_code"],
        "nutrient": row["nutrient"],
        "model_unit": model_unit,
        "input_unit": unit,
        "label_claim_input": label_claim,
        "label_claim_model_unit_per_serving": round(converted_label_claim, 6),
        "label_claim_model_unit_per_day": round(label_claim_per_day, 6),
        "predicted_percent_difference_per_serving": round(percent_diff, 6),
        "predicted_amount_per_serving": round(predicted_per_serving, 6),
        "predicted_amount_per_day": round(predicted_per_day, 6),
        "servings_per_day": servings_per_day,
        "model_source": f"DSID-4 regression table: {row['category_code']}",
        "supported_range": {
            "min_label_claim": min_claim,
            "max_label_claim": max_claim,
            "unit": model_unit,
        },
        "coefficients": {
            "intercept": intercept,
            "linear": linear,
            "quadratic": quadratic,
        },
    }


async def fdc_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    api_key = get_settings()["fdc_api_key"]
    if not api_key:
        raise HTTPException(status_code=500, detail="FDC_API_KEY is not set on the server.")

    merged_params = dict(params or {})
    merged_params["api_key"] = api_key

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
        response = await client.get(f"{FDC_BASE_URL}{path}", params=merged_params)

    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail={"message": "FoodData Central request failed", "fdc_response": response.text},
        )
    return response.json()


async def dsld_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    api_key = get_settings()["dsld_api_key"]
    if not api_key:
        raise HTTPException(status_code=500, detail="DSLD_API_KEY is not set on the server.")

    merged_params = dict(params or {})
    merged_params["api_key"] = api_key

    async with httpx.AsyncClient(
        timeout=DEFAULT_TIMEOUT_SECONDS,
        follow_redirects=True,
    ) as client:
        response = await client.get(f"{DSLD_BASE_URL}{path}", params=merged_params)

    content_type = response.headers.get("content-type", "")
    text_preview = response.text[:500]

    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail={
                "message": "DSLD request failed",
                "status_code": response.status_code,
                "content_type": content_type,
                "response_preview": text_preview,
                "url": str(response.url),
            },
        )

    if "application/json" not in content_type.lower():
        raise HTTPException(
            status_code=502,
            detail={
                "message": "DSLD returned non-JSON response",
                "content_type": content_type,
                "response_preview": text_preview,
                "url": str(response.url),
            },
        )

    try:
        return response.json()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Failed to decode DSLD JSON response",
                "content_type": content_type,
                "response_preview": text_preview,
                "url": str(response.url),
                "error": str(exc),
            },
        )


async def search_fdc_foods(query: str, page_size: int = 10, page_number: int = 1) -> Dict[str, Any]:
    api_key = get_settings()["fdc_api_key"]
    if not api_key:
        raise HTTPException(status_code=500, detail="FDC_API_KEY is not set on the server.")

    payload = {
        "query": query,
        "pageSize": page_size,
        "pageNumber": page_number,
        "dataType": DEFAULT_FDC_DATA_TYPES,
        "sortBy": "dataType.keyword",
        "sortOrder": "asc",
    }

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{FDC_BASE_URL}/foods/search",
            params={"api_key": api_key},
            json=payload,
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail={"message": "FoodData Central search failed", "fdc_response": response.text},
        )
    return response.json()


async def get_food_details(fdc_id: int) -> Dict[str, Any]:
    data = await fdc_get(f"/food/{fdc_id}", params={"format": "full"})
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Unexpected FoodData Central response format.")
    return data


def extract_food_nutrients(food_data: Dict[str, Any], grams: float = 100.0) -> List[Dict[str, Any]]:
    scaling_factor = grams / 100.0
    results: List[Dict[str, Any]] = []

    for item in food_data.get("foodNutrients", []):
        nutrient_info = item.get("nutrient") or {}
        nutrient_name = nutrient_info.get("name") or item.get("nutrientName")
        amount = item.get("amount")
        unit_name = nutrient_info.get("unitName") or item.get("unitName")

        if nutrient_name is None or amount is None or unit_name is None:
            continue

        results.append(
            {
                "nutrient": nutrient_name,
                "amount_per_100g": float(amount),
                "amount_for_grams": round(float(amount) * scaling_factor, 6),
                "unit": str(unit_name).lower(),
            }
        )

    return results


def accumulate_nutrient(
    totals: Dict[str, Dict[str, Any]],
    canonical_name: str,
    amount: float,
    unit: str,
    source: Literal["food", "supplement"],
) -> None:
    key = normalize_text(canonical_name)
    nutrient_info = resolve_canonical_nutrient(canonical_name)
    canonical = nutrient_info["canonical"]
    canonical_unit = nutrient_info["unit"] or normalize_unit_token(unit)

    try:
        converted_amount = convert_amount(
            amount,
            unit,
            canonical_unit,
            nutrient_name=canonical_name,
        )
    except Exception:
        return

    if key not in totals:
        totals[key] = {
            "nutrient": canonical,
            "unit": canonical_unit,
            "from_food": 0.0,
            "from_supplement": 0.0,
            "total": 0.0,
        }

    bucket = totals[key]
    if source == "food":
        bucket["from_food"] += converted_amount
    else:
        bucket["from_supplement"] += converted_amount

    bucket["total"] = bucket["from_food"] + bucket["from_supplement"]


def match_fdc_nutrients_to_canonical(food_nutrients: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []

    for item in food_nutrients:
        food_name_norm = normalize_text(item["nutrient"])
        for nutrient_data in NUTRIENT_SYNONYMS.values():
            fdc_names = [normalize_text(name) for name in nutrient_data["fdc_names"]]
            if food_name_norm in fdc_names:
                matched.append(
                    {
                        "canonical_nutrient": nutrient_data["canonical"],
                        "amount": item["amount_for_grams"],
                        "unit": item["unit"],
                        "original_fdc_name": item["nutrient"],
                    }
                )
                break

    return matched


@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    return HealthResponse(status="ok", app=APP_TITLE, version=APP_VERSION)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", app=APP_TITLE, version=APP_VERSION)


@app.get("/categories")
async def list_categories() -> Dict[str, Any]:
    models = load_dsid_models()
    categories = sorted(models["category_code"].dropna().astype(str).unique().tolist())
    return {"categories": categories}


@app.get("/nutrients")
async def list_nutrients(category: Optional[str] = None) -> Dict[str, Any]:
    models = load_dsid_models()
    if category:
        models = models[models["category_norm"] == normalize_text(category)]
    nutrients = sorted(models["nutrient"].dropna().astype(str).unique().tolist())
    return {"nutrients": nutrients}


@app.post("/predict-supplement")
async def predict_supplement(request: PredictSupplementRequest) -> Dict[str, Any]:
    return predict_from_model(
        category=request.category,
        nutrient=request.nutrient,
        label_claim=request.label_claim,
        unit=request.unit,
        servings_per_day=request.servings_per_day,
    )


@app.get("/search-food")
async def search_food(
    query: str = Query(..., min_length=2),
    page_size: int = Query(default=10, ge=1, le=25),
    page_number: int = Query(default=1, ge=1),
) -> Dict[str, Any]:
    data = await search_fdc_foods(query=query, page_size=page_size, page_number=page_number)

    foods = []
    for item in data.get("foods", []):
        foods.append(
            {
                "fdc_id": item.get("fdcId"),
                "description": item.get("description"),
                "data_type": item.get("dataType"),
                "brand_owner": item.get("brandOwner"),
                "gtin_upc": item.get("gtinUpc"),
                "serving_size": item.get("servingSize"),
                "serving_size_unit": item.get("servingSizeUnit"),
            }
        )

    return {
        "query": query,
        "total_hits": data.get("totalHits"),
        "current_page": data.get("currentPage"),
        "total_pages": data.get("totalPages"),
        "foods": foods,
    }


@app.get("/food-details/{fdc_id}")
async def food_details(fdc_id: int, grams: float = Query(default=100.0, gt=0)) -> Dict[str, Any]:
    data = await get_food_details(fdc_id)
    extracted = extract_food_nutrients(data, grams=grams)
    matched = match_fdc_nutrients_to_canonical(extracted)

    return {
        "fdc_id": data.get("fdcId"),
        "description": data.get("description"),
        "data_type": data.get("dataType"),
        "input_grams": grams,
        "matched_nutrients": matched,
        "all_nutrients": extracted,
    }


@app.post("/calculate-total-intake")
async def calculate_total_intake(request: CalculateTotalIntakeRequest) -> Dict[str, Any]:
    totals: Dict[str, Dict[str, Any]] = {}
    supplement_results: List[Dict[str, Any]] = []
    food_results: List[Dict[str, Any]] = []

    for supplement in request.supplements:
        prediction = predict_from_model(
            category=supplement.category,
            nutrient=supplement.nutrient,
            label_claim=supplement.label_claim,
            unit=supplement.unit,
            servings_per_day=supplement.servings_per_day,
        )
        supplement_results.append(prediction)
        accumulate_nutrient(
            totals=totals,
            canonical_name=prediction["nutrient"],
            amount=prediction["predicted_amount_per_day"],
            unit=prediction["model_unit"],
            source="supplement",
        )

    for food in request.foods:
        food_data = await get_food_details(food.fdc_id)
        extracted = extract_food_nutrients(food_data, grams=food.grams)
        matched = match_fdc_nutrients_to_canonical(extracted)
        food_results.append(
            {
                "fdc_id": food.fdc_id,
                "description": food_data.get("description"),
                "grams": food.grams,
                "matched_nutrients": matched,
            }
        )
        for item in matched:
            accumulate_nutrient(
                totals=totals,
                canonical_name=item["canonical_nutrient"],
                amount=item["amount"],
                unit=item["unit"],
                source="food",
            )

    sorted_totals = sorted(totals.values(), key=lambda x: x["nutrient"].lower())
    for item in sorted_totals:
        item["from_food"] = round(item["from_food"], 6)
        item["from_supplement"] = round(item["from_supplement"], 6)
        item["total"] = round(item["total"], 6)

    return {
        "summary": {
            "supplement_count": len(request.supplements),
            "food_count": len(request.foods),
            "nutrient_count": len(sorted_totals),
        },
        "supplement_results": supplement_results,
        "food_results": food_results,
        "totals": sorted_totals,
    }


@app.get("/dsld-search")
async def dsld_search(
    query: str = Query(..., min_length=1),
    page_size: int = Query(default=10, ge=1, le=25),
) -> Dict[str, Any]:
    data = await dsld_get(
        "/browse-products/",
        params={
            "method": "by_keyword",
            "q": query,
            "limit": page_size,
        },
    )

    raw_hits = data.get("hits", [])

    results = []
    for hit in raw_hits[:page_size]:
        src = hit.get("_source", {})
        results.append(
            {
                "id": hit.get("_id"),
                "name": src.get("fullName"),
                "brand": src.get("brandName"),
            }
        )

    return {
        "query": query,
        "count": len(results),
        "results": results,
    }


@app.get("/dsld-product/{product_id}")
async def dsld_product(product_id: str) -> Dict[str, Any]:
    data = await dsld_get(f"/label/{product_id}")

    if isinstance(data, dict):
        return {
            "product_id": product_id,
            "name": data.get("fullName") or data.get("product_name") or data.get("name") or data.get("title"),
            "brand": data.get("brandName") or data.get("brand_name") or data.get("brand"),
            "ingredients": data.get("ingredients", []),
            "label_statements": data.get("label_statements", []) or data.get("statements", []),
            "raw": data,
        }

    return {
        "product_id": product_id,
        "raw": data,
    }