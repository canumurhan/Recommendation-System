#########################
# İş Problemi
#########################

# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.

#########################
# Veri Seti
#########################
#Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih

#Görev 1: Veriyi Hazırlama

#Adım 1: armut_data.csv dosyasını okutunuz.

import mlxtend as mlx
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)
pd.set_option('display.width',500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: "%.1f" %x)
from mlxtend.frequent_patterns import apriori, association_rules

df_=pd.read_csv('WEEK 5 Recommendation System/armut_data.csv')
df=df_.copy()
df.head()
df.describe().T
df.isnull().sum()

#Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
#ServiceID ve CategoryID’yi "_" ile birleştirerek bu hizmetleri temsil edecek
#yeni bir değişken oluşturunuz. Elde edilmesi gereken çıktı:

df["Hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)
df.head()

# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir. Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti;
# 2017’in 10.ayında aldığı  9_4, 38_4  hizmetleri başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_"
# ile birleştirirek ID adında yeni bir değişkene atayınız.


df['CreateDate'] = pd.to_datetime(df['CreateDate'])
df['New_Date'] = df['CreateDate'].dt.to_period('M')
df["SepetID"]= df["UserId"].astype(str) + "_" + df["New_Date"].astype(str)
df.head()

#Görev 2: Birliktelik Kuralları Üretiniz ve Öneride bulununuz.
#Adım 1: Aşağıdaki gibi sepet, hizmet pivot table’i oluşturunuz.
new_df = pd.pivot_table(df, values="UserId", index="SepetID", columns="Hizmet", fill_value=0)
new_df = new_df.applymap(lambda x: 1 if x > 0 else 0)

#Adım 2: Birliktelik kurallarını oluşturunuz.
new_df= new_df.astype(bool)
frequent_hizmet = apriori(new_df, min_support=0.01, use_colnames=True)
association_rules = association_rules(frequent_hizmet, metric="support", min_threshold=0.01)

#Adım 3: arl_recommender fonksiyonunu kullanarak
#en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

sorted_rules = association_rules.sort_values("lift",ascending=False)
product_id="2_0"

recommendation_list = []
for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
            recommendation_list[0:3]

            recommendation_list