import pandas as pd
import logging
from collections import defaultdict
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    def __init__(self, db):
        self.db = db

    def compute_price_trends(
        self,
        ean: Optional[str] = None,
        brand: Optional[str] = None,
        supplier: Optional[str] = None,
        days: int = 90,
    ) -> pd.DataFrame:
        rows = self.db.get_price_history(ean=ean, brand=brand, supplier=supplier, days=days)
        if not rows:
            return pd.DataFrame(columns=["ean", "supplier", "price_net", "run_at", "brand", "description"])
        df = pd.DataFrame(rows)
        df["run_at"] = pd.to_datetime(df["run_at"])
        return df

    def compute_price_trend_alerts(self, threshold_pct: float = 5.0) -> List[Dict]:
        """Return list of (ean, supplier) pairs where price moved >= threshold_pct between last two batches."""
        rows = self.db.get_price_history(days=180)
        if not rows:
            return []
        df = pd.DataFrame(rows)
        df["run_at"] = pd.to_datetime(df["run_at"])
        alerts = []
        for (ean, supplier), grp in df.groupby(["ean", "supplier"]):
            grp = grp.sort_values("run_at")
            if len(grp) < 2:
                continue
            last = grp.iloc[-1]["price_net"]
            prev = grp.iloc[-2]["price_net"]
            if prev and prev > 0:
                pct = (last - prev) / prev * 100
                if abs(pct) >= threshold_pct:
                    alerts.append({
                        "ean": ean,
                        "supplier": supplier,
                        "brand": grp.iloc[-1].get("brand", ""),
                        "description": grp.iloc[-1].get("description", ""),
                        "prev_price": round(prev, 4),
                        "current_price": round(last, 4),
                        "pct_change": round(pct, 2),
                        "direction": "up" if pct > 0 else "down",
                    })
        return sorted(alerts, key=lambda x: abs(x["pct_change"]), reverse=True)

    def compute_stockout_risk(self, internal_data: List[Dict]) -> pd.DataFrame:
        if not internal_data:
            return pd.DataFrame()
        rows = []
        for p in internal_data:
            sales90d = p.get("sales90d", 0) or 0
            daily = sales90d / 90 if sales90d > 0 else 0
            stock = p.get("current_stock", p.get("stock", 0)) or 0
            days_cover = (stock / daily) if daily > 0 else 9999
            if days_cover < 7:
                urgency = "Critical"
            elif days_cover < 30:
                urgency = "Warning"
            else:
                urgency = "OK"
            rows.append({
                "ean": p.get("ean", ""),
                "description": p.get("description", ""),
                "brand": p.get("brand", ""),
                "current_stock": stock,
                "daily_sales": round(daily, 2),
                "days_cover": round(min(days_cover, 9999), 1),
                "urgency": urgency,
                "best_buy_price": p.get("best_buy_price"),
            })
        df = pd.DataFrame(rows)
        df = df[df["days_cover"] < 9999].copy()
        return df.sort_values("days_cover").reset_index(drop=True)

    def compute_supplier_win_rates(self) -> pd.DataFrame:
        rows = self.db.get_all_supplier_prices()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["run_at"] = pd.to_datetime(df["run_at"])
        min_prices = (
            df.groupby(["ean", "run_at"])["price_net"].min().reset_index()
        )
        min_prices.columns = ["ean", "run_at", "min_price"]
        df = df.merge(min_prices, on=["ean", "run_at"])
        df["is_winner"] = df["price_net"] <= df["min_price"] * 1.001
        df["price_index"] = df["price_net"] / df["min_price"].replace(0, float("nan"))
        win_stats = (
            df.groupby("supplier")
            .agg(
                total_competed=("ean", "count"),
                wins=("is_winner", "sum"),
                avg_price_index=("price_index", "mean"),
                sku_count=("ean", "nunique"),
            )
            .reset_index()
        )
        win_stats["win_rate_pct"] = (
            win_stats["wins"] / win_stats["total_competed"] * 100
        ).round(1)
        win_stats["avg_price_index"] = win_stats["avg_price_index"].round(3)
        df_sorted = df.sort_values(["supplier", "ean", "run_at"])
        df_sorted["price_change_pct"] = (
            df_sorted.groupby(["supplier", "ean"])["price_net"].pct_change() * 100
        )
        stability = (
            df_sorted.groupby("supplier")["price_change_pct"].std().reset_index()
        )
        stability.columns = ["supplier", "price_std_pct"]
        win_stats = win_stats.merge(stability, on="supplier", how="left")
        win_stats["price_std_pct"] = win_stats["price_std_pct"].fillna(0)
        win_stats["price_stability"] = win_stats["price_std_pct"].apply(
            lambda x: "High" if x < 2 else ("Medium" if x < 5 else "Low")
        )
        return win_stats.sort_values("win_rate_pct", ascending=False).reset_index(drop=True)

    def compute_brand_health(self, internal_data: List[Dict]) -> pd.DataFrame:
        if not internal_data:
            return pd.DataFrame()
        history = self.db.get_price_history(days=90)
        brand_products: Dict[str, List[Dict]] = defaultdict(list)
        for p in internal_data:
            brand = (p.get("brand") or "Unknown").strip() or "Unknown"
            brand_products[brand].append(p)
        # Latest EANs for coverage
        latest_eans: set = set()
        trend_map: Dict[str, float] = {}
        if history:
            df_h = pd.DataFrame(history)
            df_h["run_at"] = pd.to_datetime(df_h["run_at"])
            latest_batch = df_h["run_at"].max()
            latest_eans = set(df_h[df_h["run_at"] == latest_batch]["ean"].unique())
            min_per_batch = (
                df_h.groupby(["ean", "run_at"])["price_net"].min().reset_index()
            )
            for ean, grp in min_per_batch.groupby("ean"):
                grp = grp.sort_values("run_at")
                if len(grp) >= 2:
                    last = grp.iloc[-1]["price_net"]
                    prev = grp.iloc[-2]["price_net"]
                    if prev and prev > 0:
                        trend_map[ean] = (last - prev) / prev * 100
        rows = []
        for brand, products in brand_products.items():
            sku_count = len(products)
            brand_eans = {p.get("ean") for p in products}
            covered = len(brand_eans & latest_eans)
            coverage_pct = (covered / sku_count * 100) if sku_count > 0 else 0
            at_risk = 0
            for p in products:
                sales90d = p.get("sales90d", 0) or 0
                if sales90d > 0:
                    daily = sales90d / 90
                    stock = p.get("current_stock", p.get("stock", 0)) or 0
                    if stock / daily < 30:
                        at_risk += 1
            trends = [
                trend_map[p.get("ean")]
                for p in products
                if p.get("ean") in trend_map
            ]
            avg_trend = sum(trends) / len(trends) if trends else 0
            top = max(products, key=lambda p: p.get("sales90d", 0))
            rows.append({
                "brand": brand,
                "sku_count": sku_count,
                "price_trend_pct": round(avg_trend, 2),
                "at_risk_skus": at_risk,
                "coverage_pct": round(coverage_pct, 1),
                "top_opportunity": top.get("description") or top.get("ean", ""),
            })
        df = pd.DataFrame(rows)
        return df.sort_values("at_risk_skus", ascending=False).reset_index(drop=True)
