import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agent.tools import get_latest_price, convert_price, get_analytics_report


# --- get_latest_price ---

@pytest.mark.asyncio
async def test_get_latest_price_success():
    """Returns correct price string when fruit exists in DB."""
    mock_row = MagicMock()
    mock_row.Name = "Apple"
    mock_row.Avarage_Price = 2.50
    mock_row.Date = "2024-01-15"

    with patch("agent.tools.get_engine") as mock_engine:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = mock_row
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn

        result = await get_latest_price.ainvoke({"fruit": "Apple"})

    assert "Apple" in result
    assert "2.5" in result
    assert "2024-01-15" in result
    assert "PLN" in result


@pytest.mark.asyncio
async def test_get_latest_price_not_found():
    """Returns not found message when fruit doesn't exist."""
    with patch("agent.tools.get_engine") as mock_engine:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn

        result = await get_latest_price.ainvoke({"fruit": "Dragon Fruit"})

    assert "No data found" in result
    assert "Dragon Fruit" in result


# --- convert_price ---

@pytest.mark.asyncio
async def test_convert_price_success():
    """Returns correct converted price."""
    mock_row = MagicMock()
    mock_row.Name = "Apple"
    mock_row.Avarage_Price = 4.0
    mock_row.Date = "2024-01-15"

    mock_rates = [{"code": "USD", "mid": 4.0}]

    with patch("agent.tools.get_engine") as mock_engine, \
         patch("agent.tools._get_currency_rates", new_callable=AsyncMock) as mock_rates_fn:

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = mock_row
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        mock_rates_fn.return_value = mock_rates

        result = await convert_price.ainvoke({
            "fruit": "Apple",
            "target_currency": "USD"
        })

    assert "1.0" in result
    assert "USD" in result


@pytest.mark.asyncio
async def test_convert_price_invalid_currency():
    """Returns error message for unknown currency."""
    mock_row = MagicMock()
    mock_row.Avarage_Price = 4.0
    mock_row.Date = "2024-01-15"

    with patch("agent.tools.get_engine") as mock_engine, \
         patch("agent.tools._get_currency_rates", new_callable=AsyncMock) as mock_rates_fn:

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = mock_row
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        mock_rates_fn.return_value = [{"code": "USD", "mid": 4.0}]

        result = await convert_price.ainvoke({
            "fruit": "Apple",
            "target_currency": "XYZ"
        })

    assert "not found" in result


# --- get_analytics_report ---

@pytest.mark.asyncio
async def test_get_analytics_report_success():
    """Returns CSV string with report data."""
    import pandas as pd

    mock_df = pd.DataFrame({
        "Date": ["2023-01-15", "2023-06-15"],
        "Name": ["Apple", "Apple"],
        "Avarage_Price": [2.50, 3.00]
    })

    with patch("agent.tools.get_engine"), \
         patch("pandas.read_sql", return_value=mock_df):

        result = await get_analytics_report.ainvoke({
            "fruits": ["Apple"],
            "aggregation": "yearly",
            "start_year": 2023,
            "end_year": 2023
        })

    assert "Period" in result
    assert "Apple" in result


@pytest.mark.asyncio
async def test_get_analytics_report_empty():
    """Returns no data message when DB returns empty result."""
    import pandas as pd

    empty_df = pd.DataFrame(columns=["Date", "Name", "Avarage_Price"])

    with patch("agent.tools.get_engine"), \
         patch("pandas.read_sql", return_value=empty_df):

        result = await get_analytics_report.ainvoke({
            "fruits": ["Apple"],
            "aggregation": "yearly"
        })

    assert "No data found" in result