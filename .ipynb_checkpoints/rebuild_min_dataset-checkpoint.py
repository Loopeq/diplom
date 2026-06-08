from pathlib import Path
import pandas as pd

base = Path('/mnt/data/rostat_work/rostat/csv')
out_csv = Path('/mnt/data/demand_primary_training_minimal.csv')
out_dict = Path('/mnt/data/demand_primary_training_minimal_dictionary.csv')
out_summary = Path('/mnt/data/demand_primary_training_minimal_summary.txt')

# Source-derived quarterly tables from the uploaded Rosstat archive.
product = pd.read_csv(base / 'product_quarterly_dataset_no_services.csv')
stock_value = pd.read_csv(base / 'stock_value_quarterly.csv')[['period_end', 'stock_value_mean_thousand_rub']]

# Keep the main product groups used in the diploma dataset and remove the catch-all G99 group.
main_groups = sorted(set(product['product_group_code'].dropna()) - {'G99'})
df = product[product['product_group_code'].isin(main_groups)].copy()

# Join stock value, because stock days and stock value are separate source files.
df = df.merge(stock_value, on='period_end', how='left')

# Recalculate the basic demand target from the source quarterly sales and consumer-price indices.
df['sales_index'] = df['sales_value_index']
df['consumer_price_index'] = df['consumer_price_index']
df['demand_raw'] = df['sales_index'] / df['consumer_price_index'] * 100

def rebase_to_first_available(s: pd.Series) -> pd.Series:
    non_null = s.dropna()
    if non_null.empty:
        return pd.Series([pd.NA] * len(s), index=s.index)
    return s / non_null.iloc[0] * 100

df = df.sort_values(['sales_product_code', 'period_end']).reset_index(drop=True)
df['target_demand_index'] = df.groupby('sales_product_code', group_keys=False)['demand_raw'].apply(rebase_to_first_available)

# Training-minimum schema: identifiers + time keys + target + only basic source/exogenous indicators.
result = df.rename(columns={
    'sales_product_code': 'product_code',
    'sales_product_name': 'product_name',
    'product_group_code': 'group_code',
    'product_group_name': 'group_name',
    'per_capita_income_index': 'income_index',
})[[
    'period_end',
    'year',
    'quarter',
    'product_code',
    'product_name',
    'group_code',
    'group_name',
    'sales_quarter_thousand_rub',
    'sales_index',
    'consumer_price_index',
    'target_demand_index',
    'income_index',
    'stock_days_mean',
    'stock_value_mean_thousand_rub',
]]

# Primary-training dataset should not contain rows without target or base exogenous fields.
required = [
    'sales_quarter_thousand_rub',
    'sales_index',
    'consumer_price_index',
    'target_demand_index',
    'income_index',
    'stock_days_mean',
    'stock_value_mean_thousand_rub',
]
result = result.dropna(subset=required).copy()

# Round numeric values to a practical precision for CSV while preserving enough accuracy for modeling.
num_cols = result.select_dtypes(include='number').columns
result[num_cols] = result[num_cols].round(6)

# Save with BOM for correct Russian text opening in Excel.
result.to_csv(out_csv, index=False, encoding='utf-8-sig')

# Compact data dictionary.
dictionary = pd.DataFrame([
    ('period_end', 'Дата окончания квартала.'),
    ('year', 'Год наблюдения.'),
    ('quarter', 'Квартал наблюдения: 1-4. Это календарный ключ, не сезонный коэффициент.'),
    ('product_code', 'Код товарной позиции из квартального набора продаж.'),
    ('product_name', 'Название товарной позиции.'),
    ('group_code', 'Код укрупненной товарной группы. G99 исключена как прочее.'),
    ('group_name', 'Название укрупненной товарной группы.'),
    ('sales_quarter_thousand_rub', 'Квартальные продажи, тыс. руб.; исходная база для расчета индекса продаж.'),
    ('sales_index', 'Индекс квартальных продаж по товару, первый доступный квартал товара = 100.'),
    ('consumer_price_index', 'Индекс потребительских цен по соответствующей товарной позиции/группе.'),
    ('target_demand_index', 'Целевая переменная: индекс расчетного спроса, пересчитан как rebased((sales_index / consumer_price_index) * 100), первый доступный квартал товара = 100.'),
    ('income_index', 'Индекс среднедушевых денежных доходов населения.'),
    ('stock_days_mean', 'Средний уровень товарных запасов за квартал, дней.'),
    ('stock_value_mean_thousand_rub', 'Средний стоимостной уровень товарных запасов за квартал, тыс. руб.'),
], columns=['column', 'description'])
dictionary.to_csv(out_dict, index=False, encoding='utf-8-sig')

# Summary for QA.
coverage = result.groupby(['group_code', 'group_name']).agg(
    products=('product_code', 'nunique'),
    rows=('product_code', 'size'),
    first_period=('period_end', 'min'),
    last_period=('period_end', 'max'),
).reset_index()
missing = result.isna().sum()
summary_lines = []
summary_lines.append('Файл: demand_primary_training_minimal.csv')
summary_lines.append(f'Строк: {len(result)}')
summary_lines.append(f'Столбцов: {result.shape[1]}')
summary_lines.append(f'Товаров: {result.product_code.nunique()}')
summary_lines.append(f'Период: {result.period_end.min()} — {result.period_end.max()}')
summary_lines.append('Исключено: G99/Прочее, сезонные коэффициенты, sin/cos квартала, лаги, темпы роста qoq/yoy, логи, расширенный/сезонно скорректированный спрос.')
summary_lines.append('\nПокрытие по группам:')
summary_lines.append(coverage.to_string(index=False))
summary_lines.append('\nПропуски по столбцам:')
summary_lines.append(missing.to_string())
out_summary.write_text('\n'.join(summary_lines), encoding='utf-8')

print(out_csv)
print(out_dict)
print(out_summary)
print('\n'.join(summary_lines[:6]))
print('\n', coverage.to_string(index=False))
