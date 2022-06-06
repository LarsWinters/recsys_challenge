import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# path to recsys data
PATH = "data"

candidate_items = pd.read_csv(PATH + '/candidate_items.csv')
item_features = pd.read_csv(PATH + '/item_features.csv')
train_purchases = pd.read_csv(PATH + '/train_purchases.csv')
train_sessions = pd.read_csv(PATH + '/train_sessions.csv')

def EDA():
    # check for null values
    dfList = [candidate_items, item_features, train_sessions, train_purchases]
    # date object to datetime
    train_purchases['date'] = pd.to_datetime(train_purchases['date'])
    train_sessions['date'] = pd.to_datetime(train_sessions['date'])
    for i in range(len(dfList)):
        print("Info on dataframe: ")
        print()
        print(dfList[i].isnull().sum())
        print()
        print(dfList[i].info())
        print()
        print(dfList[i].shape)
        print()
    return

def exploreItemFeatures():
    # unique feature categories
    count_feature_categories = len(sorted(item_features['feature_category_id'].unique()))
    print(count_feature_categories, "unique feature categories")


    # features per item + distribution
    features_per_item = item_features.groupby(item_features['item_id'])['feature_category_id'].count().reset_index(name ='count')
    print(features_per_item)
    print("Average amount of feature categories per item: ",features_per_item['count'].mean())
    count_features_per_item = features_per_item['count'].value_counts()[:10].rename_axis('unique_values').reset_index(name='counts')
    print(count_features_per_item)
    plt.figure()
    sns.barplot(count_features_per_item['unique_values'], count_features_per_item['counts'], alpha=0.8,
                order=count_features_per_item.sort_values('counts', ascending=False).unique_values)
    plt.title('Distribution of number of features per item')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Amount of Features per Item', fontsize=12)
    plt.show()

    # generate feature matrix to get item similarity later on
    feature_matrix = pd.crosstab(index=item_features['item_id'], columns=item_features['feature_category_id'], values=item_features['feature_value_id'], aggfunc='sum')
    print("Fill NA with 0")
    feature_matrix.fillna(0, inplace=True)
    print(feature_matrix)
    feature_matrix.shape
    return

def exploreSessions():
    # items per session + distribution
    items_per_session = train_sessions.groupby(train_sessions['session_id'])['item_id'].count().reset_index(
        name='count')
    print(items_per_session)
    print()
    print("Average amount of items per session: ", items_per_session['count'].mean())
    print()
    count_items_per_session = items_per_session['count'].value_counts()[:10].rename_axis('unique_values').reset_index(
        name='counts')
    print(count_items_per_session)
    plt.figure()
    sns.barplot(count_items_per_session['unique_values'], count_items_per_session['counts'], alpha=0.8,
                order=count_items_per_session.sort_values('counts', ascending=False).unique_values)
    plt.title('Distribution of items per session')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Amount of items per session', fontsize=12)
    plt.show()
    return

def explorePurchases():
    # items per session + distribution
    purchases_per_session = train_sessions.groupby(train_sessions['session_id'])['item_id'].count().reset_index(
        name='count')
    print(purchases_per_session)
    print()
    print("Average amount of items per session: ", purchases_per_session['count'].mean())
    print()
    count_purchases_per_session = purchases_per_session['count'].value_counts()[:10].rename_axis('unique_values').reset_index(
        name='counts')
    print(count_purchases_per_session)
    plt.figure()
    sns.barplot(count_purchases_per_session['unique_values'], count_purchases_per_session['counts'], alpha=0.8,
                order=count_purchases_per_session.sort_values('counts', ascending=False).unique_values)
    plt.title('Distribution of purchases per session')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Amount of purchases per session', fontsize=12)
    plt.show()
    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #EDA()
    #exploreItemFeatures()
    #exploreSessions()
    #explorePurchases()
    print(train_sessions.shape)
    print(train_sessions.nunique())
    print(train_purchases.shape)
    final = pd.merge(train_sessions, train_purchases, on="session_id", how="left")
    print(final.shape)
    print(final.isnull().sum())
    print(final.head())
    final = final.groupby(final['session_id'])['item_id_x'].count().reset_index(name='count')
    print("Mean amount of items per session: ", final['count'].mean())
    print("Median amount of items per session: ", final['count'].median())
    print(candidate_items.nunique())
    print(item_features.nunique())
    test = item_features.groupby('item_id', as_index=False)[['feature_category_id','feature_value_id']].count()


