import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind, f_oneway
from scipy.stats import f
df = pd.read_csv('big5_standard_stats.csv', header=[0,1])
flat = []
for lvl0, lvl1 in df.columns:
    if lvl0.lower() == 'url' or 'url' in lvl1.lower():
        flat.append('url')
    elif 'Unnamed' in str(lvl1):
        flat.append(lvl0.strip())
    else:
        flat.append(f"{lvl0.strip()}_{lvl1.strip()}")
df.columns = flat
url_col = [c for c in df.columns if c == 'url'][0]
df['Команда'] = (
    df[url_col]
      .str.split('/')
      .str[-1]
      .str.replace('-Stats','', regex=False)
      .str.replace('-', ' ', regex=False)
      .str.strip()
)
metrics_df = df.drop(columns=[url_col]).copy()
cols = ['Команда'] + [c for c in metrics_df.columns if c != 'Команда']
metrics_df = metrics_df[cols]

def make_df(parsed_list):
    return pd.DataFrame(parsed_list,
         columns=["Pos","Team","Pld","W","D","L","GF","GA","GD","Pts"])

pl_parsed = [
    (1, "Manchester City", 38,28,7,3,96,34,62,91),
    (2, "Arsenal", 38,28,5,5,91,29,62,89),
    (3, "Liverpool", 38,24,10,4,86,41,45,82),
    (4, "Aston Villa", 38,20,8,10,76,61,15,68),
    (5, "Tottenham Hotspur", 38,20,6,12,74,61,13,66),
    (6, "Chelsea", 38,18,9,11,77,63,14,63),
    (7, "Newcastle United", 38,18,6,14,85,62,23,60),
    (8, "Manchester United", 38,18,6,14,57,58,-1,60),
    (9, "West Ham United", 38,14,10,14,60,74,-14,52),
    (10, "Crystal Palace", 38,13,10,15,57,58,-1,49),
    (11, "Brighton and Hove Albion", 38,12,12,14,55,62,-7,48),
    (12, "Bournemouth", 38,13,9,16,54,67,-13,48),
    (13, "Fulham", 38,13,8,17,55,61,-6,47),
    (14, "Wolverhampton Wanderers", 38,13,7,18,50,65,-15,46),
    (15, "Everton", 38,13,9,16,40,51,-11,40),
    (16, "Brentford", 38,10,9,19,56,65,-9,39),
    (17, "Nottingham Forest", 38,9,9,20,49,67,-18,32),
    (18, "Luton Town", 38,6,8,24,52,85,-33,26),
    (19, "Burnley", 38,5,9,24,41,78,-37,24),
    (20, "Sheffield United", 38,3,7,28,35,104,-69,16)
]
la_parsed = [
    (1, "Real Madrid", 38,29,8,1,87,26,61,95),
    (2, "Barcelona", 38,26,7,5,79,44,35,85),
    (3, "Girona", 38,25,6,7,85,46,39,81),
    (4, "Atletico Madrid", 38,24,4,10,70,43,27,76),
    (5, "Athletic Club", 38,19,11,8,61,37,24,68),
    (6, "Real Sociedad", 38,16,12,10,51,39,12,60),
    (7, "Real Betis", 38,14,15,9,48,45,3,57),
    (8, "Villarreal", 38,14,11,13,65,65,0,53),
    (9, "Valencia", 38,13,10,15,40,45,-5,49),
    (10, "Alaves", 38,12,10,16,36,46,-10,46),
    (11, "Osasuna", 38,12,9,17,45,56,-11,45),
    (12, "Getafe", 38,10,13,15,42,54,-12,43),
    (13, "Celta Vigo", 38,10,11,17,46,57,-11,41),
    (14, "Sevilla", 38,10,11,17,48,54,-6,41),
    (15, "Mallorca", 38,8,16,14,33,44,-11,40),
    (16, "Rayo Vallecano", 38,8,14,16,33,48,-15,38),
    (17, "Las Palmas", 38,8,12,18,33,52,-19,36),
    (18, "Cadiz", 38,6,15,17,26,55,-29,33),
    (19, "Almeria", 38,3,12,23,43,75,-32,21),
    (20, "Granada", 38,4,9,25,38,79,-41,21)
]
sa_parsed = [
    (1, "Internazionale", 38,29,7,2,89,22,67,94),
    (2, "Milan", 38,22,9,7,76,49,27,75),
    (3, "Juventus", 38,19,14,5,54,31,23,71),
    (4, "Atalanta", 38,21,6,11,72,42,30,69),
    (5, "Bologna", 38,18,14,6,54,32,22,68),
    (6, "Roma", 38,18,9,11,65,46,19,63),
    (7, "Lazio", 38,18,7,13,49,39,10,61),
    (8, "Fiorentina", 38,17,9,12,61,46,15,60),
    (9, "Torino", 38,13,14,11,36,36,0,53),
    (10, "Napoli", 38,13,14,11,55,48,7,53),
    (11, "Genoa", 38,12,13,13,45,45,0,49),
    (12, "Monza", 38,11,12,15,39,51,-12,45),
    (13, "Hellas Verona", 38,9,11,18,38,51,-13,38),
    (14, "Lecce", 38,8,14,16,32,54,-22,38),
    (15, "Cagliari", 38,8,12,18,42,68,-26,36),
    (16, "Empoli", 38,9,9,20,29,54,-25,36),
    (17, "Udinese", 38,6,19,13,37,53,-16,37),
    (18, "Frosinone", 38,8,11,19,44,69,-25,35),
    (19, "Sassuolo", 38,7,9,22,43,75,-32,30),
    (20, "Salernitana", 38,2,11,25,32,81,-49,17)
]
bl_parsed = [
    (1, "Bayer Leverkusen", 34,28,6,0,89,24,65,90),
    (2, "Stuttgart", 34,23,4,7,78,39,39,73),
    (3, "Bayern Munich", 34,23,3,8,94,45,49,72),
    (4, "RB Leipzig", 34,19,8,7,77,39,38,65),
    (5, "Dortmund", 34,18,9,7,68,43,25,63),
    (6, "Eintracht Frankfurt", 34,11,14,9,51,50,1,47),
    (7, "Heidenheim", 34,10,12,12,50,55,-5,42),
    (8, "Werder Bremen", 34,11,9,14,48,54,-6,42),
    (9, "Augsburg", 34,10,9,15,52,60,-8,39),
    (10, "Freiburg", 34,11,6,17,45,63,-18,39),
    (11, "Hoffenheim", 34,10,9,15,54,66,-12,39),
    (12, "Union Berlin", 34,9,6,19,33,58,-25,33),
    (13, "Wolfsburg", 34,9,7,18,41,56,-15,34),
    (14, "Bochum", 34,7,12,15,42,74,-32,33),
    (15, "Mainz 05", 34,7,14,13,39,50,-11,35),
    (16, "Köln", 34,5,12,17,28,60,-32,27),
    (17, "Darmstadt", 34,3,8,23,30,86,-56,17),
    (18, "Monchengladbach", 34,7,13,14,56,64,-8,34)
]
lg_parsed = [
    (1, "Paris Saint Germain", 34,22,10,2,81,33,48,76),
    (2, "Monaco", 34,20,7,7,68,42,26,67),
    (3, "Brest", 34,17,10,7,53,34,19,61),
    (4, "Lille", 34,16,11,7,52,34,18,59),
    (5, "Nice", 34,15,10,9,40,29,11,55),
    (6, "Lyon", 34,16,5,13,49,55,-6,53),
    (7, "Lens", 34,14,9,11,45,37,8,51),
    (8, "Marseille", 34,13,11,10,52,41,11,50),
    (9, "Reims", 34,13,8,13,42,48,-6,47),
    (10, "Rennes", 34,12,10,12,53,46,7,46),
    (11, "Toulouse", 34,10,11,13,42,48,-6,41),
    (12, "Montpellier", 34,9,11,14,43,50,-7,38),
    (13, "Strasbourg", 34,9,11,14,38,50,-12,38),
    (14, "Le Havre", 34,7,13,14,35,45,-10,34),
    (15, "Metz", 34,8,5,21,35,58,-23,29),
    (16, "Lorient", 34,6,10,18,43,66,-23,28),
    (17, "Clermont Foot", 34,5,11,18,26,60,-34,26),
    (18, "Nantes", 34,7,7,20,30,55,-25,28)
]
final_df = pd.concat([
    make_df(pl_parsed),
    make_df(la_parsed),
    make_df(sa_parsed),
    make_df(bl_parsed),
    make_df(lg_parsed)
], ignore_index=True)

final_df = final_df.rename(columns={
    'Team':'Команда','Pld':'Игр','W':'Выигрышей','D':'Ничьих',
    'L':'Поражений','GF':'Забито','GA':'Пропущено',
    'GD':'Разница_голов','Pts':'Очки'
})

name_corrections = {
    'ac milan':'milan','inter milan':'internazionale',
    'borussia dortmund':'dortmund','brighton':'brighton and hove albion',
    'wolverhampton':'wolverhampton wanderers','mainz':'mainz 05',
    'paris saint-germain':'paris saint germain','gladbach':'monchengladbach',
    'verona':'hellas verona'
}
def normalize(df, col='Команда'):
    df = df.copy()
    df[col] = (
        df[col].str.lower()
               .str.replace(r"\s+"," ", regex=True)
               .str.strip()
               .replace(name_corrections)
    )
    return df

metrics_norm = normalize(metrics_df, 'Команда')
final_norm   = normalize(final_df, 'Команда')


common = set(metrics_norm['Команда']).intersection(final_norm['Команда'])
m1 = metrics_norm[metrics_norm['Команда'].isin(common)]
m2 = final_norm[final_norm['Команда'].isin(common)]
merged = pd.merge(m1, m2, on='Команда', how='inner')
merged['Команда'] = merged['Команда'].str.title()
print(f"Команд после объединения: {merged.shape[0]}")
rename_dict = {
    'Pts': 'Очки',
    'GD': 'Разница_голов',
    'GF': 'Забито',
    'Poss': 'Владение_мячом',
    'Progression_PrgP': 'Прогрессия_по_передачам',
    'Progression_PrgC': 'Прогрессия_по_движениям',
    'Performance_G+A': 'Голы_и_ассисты',
    'Performance_Gls': 'Голы',
    'Performance_G-PK': 'Голы_без_пенальти',
    'Performance_Ast': 'Ассисты',
    'Expected_xG': 'Ожидаемые_голы',
    'Expected_xAG': 'Ожидаемые_ассисты',
    'Expected_npxG+xAG': 'Ожидаемые_голы_без_пенальти_и_ассисты',
    'Expected_npxG': 'Ожидаемые_голы_без_пенальти',  # нет русского аналога, можно оставить как есть
    'Per 90 Minutes_G+A': 'Голы_и_ассисты_за_90_мин',
    'Per 90 Minutes_Ast': 'Ассисты_за_90_мин',
    'Per 90 Minutes_G+A-PK': 'Голы_и_ассисты_без_пенальти_за_90_мин',
    'Per 90 Minutes_Gls': 'Голы_за_90_мин',
    'Per 90 Minutes_G-PK': 'Голы_без_пенальти_за_90_мин',
    'Per 90 Minutes_xG+xAG': 'Ожидаемые_голы_и_ассисты_за_90_мин',
    'Per 90 Minutes_xG': 'Ожидаемые_голы_за_90_мин',
    'Per 90 Minutes_xAG': 'Ожидаемые_ассисты_за_90_мин',
    'Per 90 Minutes_npxG+xAG': 'Ожидаемые_голы_и_ассисты_без_пенальти_за_90_мин',
    'Per 90 Minutes_npxG': 'Ожидаемые_голы_без_пенальти_за_90_мин',
    'Performance_PKatt': 'Пенальти_по_попыткам',
    'Performance_PK': 'Пенальти',
    'Игр': 'Игр',
    'Playing Time_Starts': 'Начало_матчей',
    'Playing Time_MP': 'Матчи_в_игре',
    'Playing Time_Min': 'Минуты_в_игре',
    'Playing Time_90s': 'Полные_90_минут',
    'Age': 'Средний возраст команды',
    'players_used': 'Использовано_игроков',
    'Performance_CrdR': 'Красные_карточки',
    'Performance_CrdY': 'Жёлтые_карточки',
    'Ничьих': 'Ничьих',
    'Пропущено': 'Пропущено',
    'Поражений': 'Поражений',
    'Pos': 'Позиция'
}

existing_rename = {k: v for k, v in rename_dict.items() if k in merged.columns}
merged = merged.rename(columns=existing_rename)


num_cols = [c for c in merged.columns if c != 'Команда' and pd.api.types.is_numeric_dtype(merged[c])]
corr_target = merged[num_cols].corr()['Выигрышей'].drop('Выигрышей')
top_corr_features = corr_target.abs().sort_values(ascending=False).head(15).index.tolist()
print("Топ признаков по корреляции с победами:")
print(corr_target[top_corr_features].sort_values(ascending=False))
plt.figure(figsize=(12,16))
sns.heatmap(merged[top_corr_features + ['Выигрышей']].corr(), annot=True, fmt='.2f', cmap='Spectral')
plt.title('Корреляционная матрица топ-15 признаков')
plt.show()
X_all =merged.drop(columns=['Команда', 'Выигрышей', 'Очки','Ничьих','Поражений'])
y = merged['Очки']
X = X_all
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
lr = LinearRegression().fit(X_train, y_train)
print("LR R^2 на тесте:", lr.score(X_test, y_test))
rf = RandomForestRegressor(random_state=42).fit(X_train, y_train)
print("RF R^2 на тесте:", rf.score(X_test, y_test))
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(8,6))
importances.plot(kind='barh')
plt.title('Важность признаков (RandomForest)')
plt.xlabel('Относительная важность')
plt.tight_layout()
plt.show()
poss_col_candidates = [c for c in merged.columns if 'Владение' in c or 'Poss' in c]
if poss_col_candidates:
    poss_col = poss_col_candidates[0]
    med_poss = merged[poss_col].median()
    group_low = merged[merged[poss_col] <= med_poss]['Выигрышей']
    group_high = merged[merged[poss_col] > med_poss]['Выигрышей']
    var_low = group_low.var(ddof=1)
    var_high = group_high.var(ddof=1)
    if var_low > var_high:
        f_stat = var_low / var_high
        dfn = len(group_low) - 1
        dfd = len(group_high) - 1
    else:
        f_stat = var_high / var_low
        dfn = len(group_high) - 1
        dfd = len(group_low) - 1
    p_value_f = 2 * min(f.cdf(f_stat, dfn, dfd), 1 - f.cdf(f_stat, dfn, dfd))
    print(f"F-тест на равенство дисперсий между группами владения мячом <= {med_poss:.2f} и > {med_poss:.2f}")
    print(f"F-статистика = {f_stat:}, p-value = {p_value_f:}")
    equal_var = p_value_f > 0.05  # если p > 0.05, дисперсии можно считать равными

    t_stat, p_value_t = ttest_ind(group_low, group_high, equal_var=equal_var)
    print(f"t-тест для сравнения средних числа выигранных матчей в группах:")
    print(f"t-статистика = {t_stat:}, p-value = {p_value_t:}")

    if p_value_t < 0.05:
        print("Различия в среднем числе выигранных матчей между группами статистически значимы.")
    else:
        print("Нет статистически значимых различий в среднем числе выигранных матчей между группами.")
else:
    print("Не найден признак владения мячом для анализа.")


X = merged.drop(columns=['Команда', 'Выигрышей', 'Очки','Ничьих','Поражений'])
y_wins = merged['Выигрышей']
y_pts = merged['Очки']
X_train, X_test, y_wins_train, y_wins_test = train_test_split(X, y_wins, test_size=0.2, random_state=12)
_, _, y_pts_train, y_pts_test = train_test_split(X, y_pts, test_size=0.2, random_state=12)

#Линейная регрессия
lr_model_wins = LinearRegression()
lr_model_pts = LinearRegression()
lr_model_wins.fit(X_train, y_wins_train)
lr_model_pts.fit(X_train, y_pts_train)
#Случайный лес
rf_model_wins = RandomForestRegressor(random_state=42)
rf_model_pts = RandomForestRegressor(random_state=42)
rf_model_wins.fit(X_train, y_wins_train)
rf_model_pts.fit(X_train, y_pts_train)

wins_preds_lr = lr_model_wins.predict(X_test)
pts_preds_lr = lr_model_pts.predict(X_test)
wins_preds_rf = rf_model_wins.predict(X_test)
pts_preds_rf = rf_model_pts.predict(X_test)

plt.figure(figsize=(14,6))
plt.subplot(1, 2, 1)
plt.scatter(y_wins_test, wins_preds_lr, label='Линейная регрессия', alpha=0.7)
plt.scatter(y_wins_test, wins_preds_rf, label='Случайный лес', alpha=0.7)
plt.plot([min(y_wins_test), max(y_wins_test)], [min(y_wins_test), max(y_wins_test)], 'k--')
plt.xlabel('Истинные выигрыши')
plt.ylabel('Прогноз выигрышей')
plt.title('Прогноз выигрышей')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(y_pts_test, pts_preds_lr, label='Линейная регрессия', alpha=0.7)
plt.scatter(y_pts_test, pts_preds_rf, label='Случайный лес', alpha=0.7)
plt.plot([min(y_pts_test), max(y_pts_test)], [min(y_pts_test), max(y_pts_test)], 'k--')
plt.xlabel('Истинные очки')
plt.ylabel('Прогноз очков')
plt.title('Прогноз очков')
plt.legend()
plt.tight_layout()
plt.show()

wins_preds_rf = lr_model_wins.predict(X_test)
pts_preds_rf = lr_model_pts.predict(X_test)
test_indices = X_test.index
team_names = merged.loc[test_indices, 'Команда'].values
real_pts = y_pts_test.values
real_wins = y_wins_test.values
results_df = pd.DataFrame({
    'Команда': team_names,
    'Реальные очки': real_wins,
    'Спрогнозированные очки': wins_preds_rf.round(1)
})
results_df = results_df.sort_values(by='Реальные очки', ascending=False).reset_index(drop=True)
print(results_df)
from prettytable import PrettyTable
table = PrettyTable()
table.field_names = ["Команда", "Реальные очки", "Спрогнозированные очки"]
for i in range(len(results_df)):
    table.add_row([
        results_df.loc[i, 'Команда'],
        results_df.loc[i, 'Реальные очки'],
        results_df.loc[i, 'Спрогнозированные очки']
    ])
print(table)
