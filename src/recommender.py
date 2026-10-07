import pandas as pd

class PropertyRecommender:
    def __init__(self, data_path='data/train.csv'):
        self.df = pd.read_csv(data_path)
        # Clean essential features for filter matching
        self.df['FullBath'] = self.df['FullBath'].fillna(1)
        self.df['BedroomAbvGr'] = self.df['BedroomAbvGr'].fillna(2)

    def recommend(self, max_budget, location=None, min_bedrooms=1, top_k=5):
        filtered = self.df[self.df['SalePrice'] <= max_budget]

        if location and location != 'All':
            filtered = filtered[filtered['Neighborhood'] == location]

        if min_bedrooms:
            filtered = filtered[filtered['BedroomAbvGr'] >= min_bedrooms]

        # Sort by overall quality and living area
        recommended = filtered.sort_values(by=['OverallQual', 'GrLivArea', 'SalePrice'], ascending=[False, False, True]).head(top_k)

        results = []
        for _, row in recommended.iterrows():
            reason = (
                f"Matches budget under ${max_budget:,.0f} in {row['Neighborhood']}. "
                f"Offers {row['BedroomAbvGr']} bedrooms, {row['GrLivArea']} sqft living area, "
                f"and high quality rating ({row['OverallQual']}/10)."
            )
            results.append({
                'id': int(row['Id']),
                'price': float(row['SalePrice']),
                'neighborhood': row['Neighborhood'],
                'bedrooms': int(row['BedroomAbvGr']),
                'sqft': int(row['GrLivArea']),
                'quality': int(row['OverallQual']),
                'explanation': reason
            })
        return results