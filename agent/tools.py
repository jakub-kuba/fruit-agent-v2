import httpx
import io
import pandas as pd
from sqlalchemy import text
from langchain_core.tools import tool
from db.engine import get_engine

@tool
async def get_latest_price(fruit: str) -> str:
    """Fetches the most recent price for a specific fruit in PLN from the database."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT Date, Name, Avarage_Price
            FROM fruit
            WHERE Name = :fruit
            AND Avarage_Price IS NOT NULL
            ORDER BY Date DESC
            LIMIT 1            
        """), {"fruit": fruit})
        row = result.fetchone()

    if not row:
        return f'No data found for fruit: "{fruit}".'
    return f'The latest price for {row.Name} is {row.Avarage_Price} PLN (Date: {row.Date}).'
    

async def _get_currency_rates() -> str:
    """Fetches live currency exchange rates from NBP API."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.nbp.pl/api/exchangerates/tables/A/?format=json"
        )
        response.raise_for_status()
        return response.json()[0]["rates"]
    

@tool
async def convert_price(fruit: str, target_currency: str) -> str:
    """Converts the latest fruit price from PLN to a specified currency."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT Date, Name, Avarage_Price
            FROM fruit
            WHERE Name = :fruit
            AND Avarage_Price IS NOT NULL
            ORDER BY Date DESC
            LIMIT 1
        """), {"fruit": fruit})
        row = result.fetchone()

    if not row:
        return f'No data found for fruit: "{fruit}".'
    

    rates = await _get_currency_rates()

    rate = next(
        (r for r in rates if r["code"].lower() == target_currency.lower()),
        None
    )

    if not rate:
        return f'Currency "{target_currency}" not found in NBP rates.'
    
    converted  = round(float(row.Avarage_Price) / rate["mid"], 2)

    return (
        f'The price of {row.Name} is {row.Avarage_Price} PLN (Date: {row.Date}), '
        f'which equals {converted} {target_currency.upper()}.'
    )


@tool
async def get_analytics_report(
    fruits: list[str],
    aggregation: str,
    start_year: int | None = None,
    end_year: int | None = None
) -> str:
    """Generates a monthly or yearly average price report for specified fruits.
    
    Args:
        fruits: List of fruit names e.g. ['Apple', 'Banana']
        aggregation: 'monthly' or 'yearly'
        start_year: Optional start year filter
        end_year: Optional end year filter
    """
    engine = get_engine()

    query = """
        SELECT Date, Name, Avarage_Price
        FROM fruit
        WHERE Name IN :fruits
        AND Avarage_Price IS NOT NULL
    """
    params = {"fruits": tuple(fruits)}

    if start_year:
        query += " AND YEAR(Date) >= :start_year"
        params["start_year"] = start_year
    if end_year:
        query += " AND YEAR(Date) <= :end_year"
        params["end_year"] = end_year

    query += " ORDER BY Date ASC"

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    if df.empty:
        return "No data found for the given criteria."

    df["Date"] = pd.to_datetime(df["Date"])
    df["Avarage_Price"] = pd.to_numeric(df["Avarage_Price"])

    # Group by period
    if aggregation == "monthly":
        df["Period"] = df["Date"].dt.to_period("M").astype(str)
    else:
        df["Period"] = df["Date"].dt.to_period("Y").astype(str)

    # Pivot table — owoce jako kolumny, okresy jako wiersze
    pivot = (
        df.groupby(["Period", "Name"])["Avarage_Price"]
        .mean()
        .round(2)
        .unstack(level="Name")
        .reset_index()
    )

    pivot.columns.name = None
    return pivot.to_csv(index=False)