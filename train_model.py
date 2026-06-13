
!pip install sentence-transformers scikit-learn --quiet

import os
import pickle
import time
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from google.colab import drive


drive.mount('/content/drive')
SAVE_PATH = "/content/drive/MyDrive/trained_intent_model.pkl"


INTENT_EXAMPLES = {

    "order_tracking": [

        # ===================== ENGLISH =====================
        "where is my order", "track my order", "order status",
        "what is the status of my order", "when will my order arrive",
        "my order has not been delivered yet", "I want to track my shipment",
        "can you check my order", "order not received", "order is delayed",
        "your order is on track", "order has been dispatched",
        "order has been shipped", "I will check your order",
        "let me check the order status", "order is out for delivery",
        "estimated delivery date", "where is my parcel",
        "give me an update on my delivery", "track order number",

        "I placed an order and want to know where it is",
        "order still processing", "order confirmation not received",
        "expected delivery", "shipment update please",
        "when does my order reach", "delivery update",
        "my package has not arrived yet",
        "I am waiting for my order",
        "can you tell me delivery status",
        "what happened to my order",
        "why is my order late",
        "is my order shipped or not",
        "order stuck somewhere",
        "check delivery progress",
        "track my package please",
        "where exactly is my shipment now",
        "can you locate my order",
        "what is the current location of my order",

        # ===================== HINGLISH =====================
        "mera order kaha hai", "order kab aayega",
        "mera order track karo na", "delivery kab hogi bhai",
        "order ka status batao", "mera parcel abhi tak nahi aaya",
        "order dispatch hua kya", "mera shipment kaha tak aaya",
        "delivery update do", "order ka update chahiye",
        "mera order pending kyu hai", "order delay kyu ho raha hai",
        "tracking number se check karo", "order kaha atka hai",
        "mera order kab milega", "order kaha pahucha hai",
        "mera order abhi tak kyu nahi aaya",
        "delivery late ho gayi hai kya",
        "order process ho gaya kya",
        "mera order confirm hua hai kya",
        "order ka live status batao",
        "shipment kaha tak aaya hai",

        # ===================== HINDI =====================
        "order kahan hai", "mera order track karo", "order status kya hai",
        "order abhi tak nahi aaya", "delivery kab hogi",
        "order dispatch hua kya", "shipment ka status batao",
        "mera parcel kahan hai", "order ki jankari chahiye",
        "mera order abhi tak deliver nahi hua",
        "order ka current status kya hai",
        "delivery me kitna time lagega",
        "order ka pata lagao",
        "order kaha pahucha hai",
        "order ki location batao",
        "mera order late ho raha hai",
        "shipment delay ho gaya hai kya",
        "delivery ki date kya hai",
        "order ka tracking detail do",

        # ===================== MARATHI =====================
        "माझी ऑर्डर कुठे आहे", "ऑर्डर स्टेटस सांगा",
        "ऑर्डर कधी येणार", "डिलिव्हरी अपडेट द्या",
        "माझं पार्सल कुठे आहे", "ऑर्डर अजून आली नाही",
        "ऑर्डर पाठवली आहे का", "शिपमेंट स्टेटस काय आहे",
        "माझी ऑर्डर ट्रॅक करा",
        "डिलिव्हरी कधी होणार आहे",
        "ऑर्डर उशीर का झाली आहे",
        "ऑर्डरची माहिती द्या",
        "माझी ऑर्डर कुठपर्यंत आली आहे",
        "ऑर्डर प्रोसेसमध्ये आहे का",
        "डिलिव्हरी लेट झाली आहे का",
        "माझं शिपमेंट कुठे आहे",

        # ===================== REAL CALL VARIATIONS =====================
        "hello my order status",
        "can you check order for me",
        "order number se status batao",
        "status batao mera order ka",
        "just checking order update",
        "i want update on my order",
        "any update on delivery",
        "pls track my order",
        "track karo order",
        "order status pls",
        "mera order update",
        "status chahiye",
        "order ka kya hua",
        "order ka scene kya hai",
        "delivery ka update kya hai",
        "mera order kaha tak aaya",
    ],

    "vehicle_inquiry": [

        # ===================== ENGLISH =====================
        "my vehicle number is", "vehicle registration number",
        "my car number is", "registration plate",
        "vehicle reg is", "vehicle no is", "my car reg is",

        "I need parts for my car", "parts for my car",
        "spare parts for my car",
        "I have a car", "my vehicle", "car registration",
        "number plate", "what parts fit my car",
        "compatible parts for my car", "which parts work for my vehicle",

        "I drive a Maruti", "I have a Honda City", "I have a Hyundai",
        "Swift spare parts", "Alto parts", "Innova parts",
        "Baleno spare parts", "WagonR parts", "Creta parts",
        "Fortuner parts", "i20 parts", "Verna parts",
        "Scorpio parts", "XUV500 parts", "Nexon parts",

        "my car model is", "vehicle make and model",
        "my vehicle is Maruti Swift", "I own a Hyundai Creta",
        "I have a Tata Nexon", "my car is Honda City",

        "need parts for my car model",
        "spare parts for my vehicle number",
        "can you check parts for my vehicle",
        "what parts are available for my car",
        "I want parts according to my vehicle number",
        "check compatibility for my car",
        "I want genuine parts for my car",
        "do you have parts for my car model",

        # ===================== HINGLISH =====================
        "meri gaadi ka number hai", "mera vehicle number hai",
        "gaadi ke parts chahiye", "meri car ke liye parts chahiye",
        "vehicle number se parts batao",
        "mere vehicle ke parts check karo",
        "meri gaadi ka model hai",
        "meri car Swift hai parts chahiye",
        "mere paas Hyundai hai parts chahiye",
        "gaadi ke hisaab se parts batao",
        "vehicle ke according parts chahiye",
        "meri gaadi ke liye compatible parts batao",
        "car ke spare parts chahiye",
        "vehicle number dalke check karo",
        "meri gaadi ka number MH12 hai",
        "vehicle number se parts milenge kya",
        "model ke hisaab se parts chahiye",
        "meri gaadi ke parts available hai kya",

        # ===================== HINDI =====================
        "meri gaadi ka number hai", "vehicle number batao",
        "gaadi ke parts chahiye", "meri car ka number",
        "vehicle registration number dena", "gaadi ka model batao",
        "meri car ke liye parts chahiye",
        "meri gaadi ke liye kaun se parts milenge",
        "mere vehicle ke liye parts batao",
        "gaadi ke hisab se parts chahiye",
        "meri gaadi ka model Swift hai",
        "vehicle number ke basis par parts batao",
        "mujhe apni car ke parts chahiye",
        "meri car ke spare parts chahiye",
        "kaun se parts fit honge meri gaadi me",
        "vehicle ke hisab se compatible parts batao",

        # ===================== MARATHI =====================
        "माझ्या गाडीचा नंबर आहे", "गाडीसाठी पार्ट हवेत",
        "माझी गाडी Maruti आहे", "गाडीचा नंबर सांगतो",
        "माझ्या कारसाठी पार्ट हवेत",
        "माझ्या गाडीचे स्पेअर पार्ट हवेत",
        "माझ्या वाहनासाठी कोणते पार्ट आहेत",
        "गाडीच्या नंबरवरून पार्ट सांगा",
        "माझ्या वाहनासाठी compatible पार्ट हवेत",
        "गाडीच्या मॉडेलनुसार पार्ट हवेत",

        # ===================== REAL CALL VARIATIONS =====================
        "my vehicle number is MH12AB1234",
        "vehicle number MH14XY5678",
        "car number is MH12 TY 1212",
        "my vehicle is MH12 something",

        "my car is Swift need parts",
        "I have one car need parts",
        "checking parts for my vehicle",

        "vehicle number bata raha hu suno",
        "gaadi ka number bol raha hu",
        "vehicle ka number note karo",
        "meri gaadi ka number suno",

        "I will tell my vehicle number",
        "my vehicle details are",
        "my car details are",

        "I want parts for this vehicle",
        "check parts for this number",
        "vehicle number ke hisab se batao",
        "gaadi ke number pe parts chahiye",
    ],

    "order_return": [

        # ===================== ENGLISH =====================
        "I want to return this", "return my order", "return this part",
        "wrong part was delivered", "I received the wrong item",
        "I need to return", "how do I return", "initiate return",
        "the part does not fit", "part is not compatible",
        "I want a refund", "when will I get my refund",
        "money back", "please process my return",
        "the part is defective", "part is damaged",
        "broken item received", "incorrect item delivered",
        "exchange this part", "I want to exchange",
        "return and refund", "refund status",

        "return pickup when", "return not initiated",
        "can I return this", "return policy",
        "part quality issue", "part not working",

        "this part is not fitting my car",
        "wrong spare part delivered",
        "I received damaged spare part",
        "part is faulty", "defective product received",
        "I want replacement", "replace this item",
        "need to return the product",
        "refund for this order",
        "return request not processed",
        "return status update",
        "my return is still pending",
        "refund not received yet",
        "why my refund is delayed",
        "return rejected why",
        "exchange request status",
        "I want to return due to quality issue",

        # ===================== HINGLISH =====================
        "return karna hai", "mujhe return karna hai",
        "galat part aaya hai", "wrong part deliver hua hai",
        "paise wapas chahiye", "refund kab milega",
        "part exchange karna hai", "part wapas karna hai",

        "ye part fit nahi ho raha",
        "galat spare part mila hai",
        "damage part mila hai",
        "part ka quality acha nahi hai",
        "ye product kharab hai",

        "return ka process batao",
        "return request dalni hai",
        "return pickup kab hoga",
        "refund delay ho raha hai",
        "refund abhi tak nahi mila",

        "exchange karna hai",
        "replacement chahiye",
        "mera return accept kyu nahi hua",
        "return status kya hai",
        "refund ka update kya hai",

        # ===================== HINDI =====================
        "return karna hai", "galat part aaya hai", "paise wapas chahiye",
        "refund kab milega", "part exchange karna hai",
        "wrong item mila", "damaged part aaya", "part wapas karna hai",
        "return policy kya hai",

        "ye part meri gaadi me fit nahi ho raha",
        "mujhe replacement chahiye",
        "product kharab hai",
        "part defective hai",
        "return ka process bataiye",
        "return kab pickup hoga",
        "refund abhi tak nahi mila",
        "refund delay ho raha hai",
        "mera return abhi tak process nahi hua",
        "return reject kyun hua",

        # ===================== MARATHI =====================
        "परत करायचे आहे", "पैसे परत करा", "चुकीचा पार्ट आला",
        "रिटर्न कसे करायचे", "परतावा कधी मिळेल",

        "हा पार्ट बसत नाही",
        "चुकीचा स्पेअर पार्ट मिळाला",
        "डॅमेज पार्ट आला",
        "प्रॉडक्ट खराब आहे",
        "मला रिटर्न करायचे आहे",

        "रिटर्न प्रोसेस सांगा",
        "रिटर्न पिकअप कधी होणार",
        "रिफंड अजून मिळाला नाही",
        "रिफंड उशीर झाला आहे",
        "माझा रिटर्न अजून प्रोसेस झाला नाही",
        "रिटर्न स्टेटस काय आहे",

        # ===================== REAL CALL VARIATIONS =====================
        "return karna hai part",
        "part wapas lena hai",
        "refund chahiye urgent",
        "wrong item mila sir",
        "part sahi nahi hai",
        "exchange kar do please",
        "return request karni hai",
        "pickup kab aayega",
        "refund kab tak milega",
        "return pending hai",
        "mera refund nahi aaya",
        "return ka kya hua",
        "refund ka kya scene hai",
        "part problem hai return karna hai",
    ],

    "invoice_request": [

        # ===================== ENGLISH =====================
        "I need my invoice", "send me the invoice", "invoice for my order",
        "bill for my purchase", "tax invoice", "GST invoice",
        "I need a receipt", "can you send the bill",
        "invoice number", "where is my invoice",
        "I need the invoice for warranty claim",
        "purchase receipt", "payment receipt",
        "invoice copy", "duplicate invoice",
        "billing statement", "itemized bill",

        "GST number on invoice", "HSN code on invoice",
        "need bill for reimbursement", "company invoice needed",
        "send invoice on email", "invoice not received",

        "please share invoice", "invoice not generated",
        "invoice missing", "invoice not available",
        "can you resend invoice", "send bill copy",
        "need invoice urgently",
        "invoice download link",
        "how to get invoice",
        "invoice for my order id",
        "bill required for accounting",
        "invoice for expense claim",
        "need GST details in invoice",

        # ===================== HINGLISH =====================
        "invoice chahiye", "bill bhejo", "receipt chahiye",
        "GST bill chahiye", "invoice send karo",
        "mujhe bill chahiye", "kaccha bill chahiye",
        "invoice email pe bhejo",

        "invoice nahi mila", "bill nahi mila",
        "invoice resend karo",
        "GST invoice bhejo",
        "invoice download kaise kare",
        "invoice urgently chahiye",
        "bill copy bhejo",
        "invoice ka link bhejo",
        "order ka invoice bhejo",
        "company ke liye invoice chahiye",
        "reimbursement ke liye bill chahiye",
        "invoice me GST number add karo",

        # ===================== HINDI =====================
        "invoice chahiye", "bill bhejo", "receipt chahiye",
        "GST bill chahiye", "invoice send karo",
        "mujhe bill chahiye",

        "invoice abhi tak nahi mila",
        "bill abhi tak nahi aaya",
        "invoice dobara bhejo",
        "GST invoice bhejiye",
        "invoice kaise milega",
        "invoice ki copy chahiye",
        "order ka bill bhejiye",
        "invoice email par bhejiye",
        "mujhe turant invoice chahiye",
        "invoice download kaise kare",

        # ===================== MARATHI =====================
        "बिल पाठवा", "इनव्हॉइस हवा", "GST बिल द्या",
        "जीएसटी इनव्हॉइस हवा",

        "मला बिल पाठवा",
        "इनव्हॉइस मिळाला नाही",
        "इनव्हॉइस पुन्हा पाठवा",
        "GST बिल पाठवा",
        "इनव्हॉइस ईमेल वर पाठवा",
        "इनव्हॉइस कसा डाउनलोड करायचा",
        "माझ्या ऑर्डरचा बिल पाठवा",
        "इनव्हॉइसची कॉपी द्या",
        "मला तातडीने इनव्हॉइस हवा",

        # ===================== REAL CALL VARIATIONS =====================
        "bill chahiye sir",
        "invoice bhej do",
        "invoice bhejna hai",
        "bill send karo please",
        "invoice ka kya hua",
        "invoice nahi aaya",
        "invoice ka update",
        "bill nahi mila abhi tak",
        "invoice resend kar do",
        "GST bill chahiye urgent",
        "invoice mail pe bhejo",
        "invoice ka copy chahiye",
        "order ka bill chahiye",
    ],

    "part_availability": [

        # ===================== ENGLISH =====================
        "is this part available", "do you have this part in stock",
        "check availability", "is it in stock", "do you sell",
        "do you have brake pads", "is the part available",
        "stock check", "part availability", "is this available",
        "do you carry this part", "do you have it",
        "part in stock", "available or not",

        "I am looking for a part", "need to check stock",
        "do you have Honda spare parts", "Maruti parts available",
        "air filter in stock", "clutch plate available",

        "when will it be back in stock", "out of stock when available",
        "do you stock genuine parts", "OEM parts available",
        "aftermarket parts available", "spare part availability",

        "is this part available for my car",
        "do you have parts for my vehicle",
        "availability for this car part",
        "is this spare part in stock",
        "check if this part is available",
        "can you confirm stock for this item",
        "do you have stock for this model",
        "is this product available right now",
        "can I order this part now",
        "is it ready to ship",
        "is this available for immediate delivery",
        "stock available for my car model",
        "do you have original parts",
        "do you have genuine spare parts",
        "is OEM part available",
        "is aftermarket part available",

        # ===================== HINGLISH =====================
        "part available hai kya", "stock mein hai kya",
        "kya yeh part milega", "part hai aapke paas",
        "stock kab aayega", "availability check karo",

        "ye part available hai kya",
        "meri car ke liye part available hai kya",
        "stock check karo please",
        "ye spare part mil jayega kya",
        "abhi stock mein hai kya",
        "part ready hai kya",
        "ye part order kar sakte hai kya",
        "ye part abhi available hai kya",
        "stock khatam ho gaya kya",
        "kab tak stock aayega",
        "genuine part available hai kya",
        "OEM part milega kya",
        "aftermarket part hai kya",

        # ===================== HINDI =====================
        "part available hai kya", "stock mein hai kya",
        "kya yeh part milega", "part hai aapke paas",
        "stock kab aayega", "availability check karo",

        "meri gaadi ke liye part available hai",
        "ye part abhi uplabdh hai kya",
        "stock ki jankari dijiye",
        "kya yeh part abhi mil sakta hai",
        "kya yeh spare part uplabdh hai",
        "kya yeh part turant mil jayega",
        "genuine part uplabdh hai kya",
        "OEM part uplabdh hai kya",

        # ===================== MARATHI =====================
        "पार्ट उपलब्ध आहे का", "स्टॉक आहे का",
        "पार्ट कधी येणार", "उपलब्धता तपासा",

        "हा पार्ट उपलब्ध आहे का",
        "माझ्या गाडीसाठी पार्ट आहे का",
        "स्टॉक मध्ये आहे का",
        "हा स्पेअर पार्ट मिळेल का",
        "स्टॉक संपला आहे का",
        "पुन्हा स्टॉक कधी येणार",
        "जेन्युइन पार्ट आहे का",
        "OEM पार्ट उपलब्ध आहे का",

        # ===================== REAL CALL VARIATIONS =====================
        "part hai kya",
        "stock hai kya",
        "available hai kya sir",
        "check kar ke batao stock",
        "ye part milega kya",
        "abhi available hai kya",
        "stock ka kya scene hai",
        "stock khatam to nahi hai",
        "order kar sakte hai kya",
        "delivery ke liye ready hai kya",
        "part available confirm karo",
        "availability batao",
        "stock check kar do",
        "ye item milega kya",
    ],

    "price_inquiry": [

        # ===================== ENGLISH =====================
        "what is the price", "how much does it cost", "price of this part",
        "cost of the part", "how much", "what is the rate",
        "pricing", "price list", "give me the price",

        "how much for brake pads", "price for oil filter",
        "cost of clutch plate", "price quote",

        "do you give discount", "any offer", "any deal",
        "best price", "cheapest price", "price comparison",

        "MRP", "selling price", "what will it cost",
        "total cost with delivery", "shipping charge",

        "bulk discount", "wholesale price", "quantity discount",
        "price including GST", "final price", "negotiable price",

        "price for genuine part", "price difference original vs local",

        "what is the cost for my car part",
        "price for this spare part",
        "can you give price details",
        "what will be the final amount",
        "how much will I have to pay",
        "give me quotation",
        "can you share price quote",
        "price for my vehicle part",
        "what is total payable amount",
        "price with GST included",
        "what is landed cost",
        "how much including delivery",
        "do you have best price offer",
        "any discount on this part",
        "is price negotiable",

        # ===================== HINGLISH =====================
        "price kya hai", "kitne ka hai", "keemat batao",
        "discount milega", "rate kya hai", "offer hai kya",
        "saste mein milega", "total amount kitna",

        "iska price batao",
        "final price kya hoga",
        "kitna padega total",
        "discount de sakte ho kya",
        "best price batao",
        "thoda kam karo price",
        "offer chal raha hai kya",
        "GST ke sath price batao",
        "delivery charge kitna hai",
        "total cost kitni hogi",
        "price compare karo",
        "genuine part ka price kya hai",
        "OEM part ka rate batao",
        "aftermarket part ka price kya hai",

        # ===================== HINDI =====================
        "price kya hai", "kitne ka hai", "keemat batao",
        "discount milega", "rate kya hai", "offer hai kya",

        "iska daam kya hai",
        "kul kitna kharcha hoga",
        "final amount kya hoga",
        "GST ke saath price bataiye",
        "delivery charge kitna lagega",
        "mujhe total price bataiye",
        "kya discount milega",
        "kya yeh sasta milega",
        "OEM part ki keemat kya hai",
        "genuine part ka daam kya hai",

        # ===================== MARATHI =====================
        "किंमत काय आहे", "किती रुपये", "सवलत मिळेल का",
        "एकूण किंमत सांगा",

        "हा पार्ट कितीला आहे",
        "फायनल प्राइस काय आहे",
        "एकूण खर्च किती होईल",
        "GST सहित किंमत सांगा",
        "डिलिव्हरी चार्ज किती आहे",
        "सवलत मिळेल का",
        "कमी दरात मिळेल का",
        "OEM पार्टची किंमत काय आहे",
        "जेन्युइन पार्ट कितीला आहे",

        # ===================== REAL CALL VARIATIONS =====================
        "price batao",
        "rate batao",
        "kitna padega",
        "final amount batao",
        "total kya hoga",
        "discount do",
        "best rate kya hai",
        "offer kya chal raha hai",
        "price ka kya scene hai",
        "kitna charge karoge",
        "iska rate kya hai",
        "bill kitna banega",
        "total bill kitna hoga",
        "price confirm karo",
        "quotation bhejo",
        "price details chahiye",
    ],

    "payment_issue": [

        # ===================== ENGLISH =====================
        "payment failed", "payment was deducted but order not placed",
        "I was charged twice", "double payment", "money deducted no order",
        "transaction failed but money gone", "payment not confirmed",
        "payment stuck", "refund for failed payment",
        "bank deducted money but order failed", "UPI payment failed",
        "net banking problem", "payment issue",

        "money debited no confirmation", "payment pending",
        "payment not reflecting", "order not placed after payment",
        "paid but no order id", "charged but no delivery",

        "my payment went through but no confirmation",
        "amount deducted but order not created",
        "payment completed but order missing",
        "transaction successful but no update",
        "why is payment not showing",
        "payment gateway error",
        "payment declined but money deducted",
        "I paid but didn’t get order confirmation",
        "payment failed but amount deducted",
        "double charge on my account",
        "I was billed twice",
        "payment not updated in system",
        "payment processing issue",
        "refund for failed transaction",
        "when will I get my money back",
        "payment reversed or not",

        # ===================== HINGLISH =====================
        "payment kat gaya order nahi hua", "double payment ho gaya",
        "paise gaye order nahi aaya", "payment problem",
        "transaction fail ho gaya paise gaye",

        "payment ho gaya par order nahi bana",
        "amount deduct ho gaya but order nahi aaya",
        "paise kat gaye confirmation nahi mila",
        "payment stuck ho gaya",
        "UPI se payment fail ho gaya",
        "net banking se problem hua",
        "card se payment kiya par issue hai",

        "do baar payment ho gaya",
        "double charge ho gaya",
        "refund kab milega payment ka",
        "payment reflect nahi ho raha",
        "order id nahi mila payment ke baad",
        "payment ka kya scene hai",
        "paise chale gaye but order nahi mila",
        "payment pending dikh raha hai",
        "payment verify karo please",

        # ===================== HINDI =====================
        "payment kat gaya order nahi hua", "double payment ho gaya",
        "paise gaye order nahi aaya", "payment problem",
        "transaction fail ho gaya paise gaye",

        "mera payment ho gaya par order nahi bana",
        "paise kat gaye par confirmation nahi mila",
        "payment pending dikha raha hai",
        "payment system me reflect nahi ho raha",
        "do baar paise kat gaye",
        "refund kab milega",
        "mera paisa wapas kab milega",
        "payment ka update bataiye",
        "order id nahi mila payment ke baad",

        # ===================== MARATHI =====================
        "पेमेंट अडले", "पैसे गेले ऑर्डर नाही",
        "दुप्पट पेमेंट झाले", "पेमेंट फेल झाले",

        "पेमेंट झाले पण ऑर्डर नाही",
        "पैसे गेले पण कन्फर्मेशन नाही",
        "पेमेंट सिस्टममध्ये दिसत नाही",
        "दोनदा पैसे गेले",
        "रिफंड कधी मिळेल",
        "पेमेंट पेंडिंग आहे",
        "पेमेंट अपडेट नाही",

        # ===================== REAL CALL VARIATIONS =====================
        "payment issue hai",
        "paise kat gaye",
        "payment fail ho gaya",
        "payment ho gaya kya",
        "confirm nahi hua payment",
        "double charge ho gaya sir",
        "payment ka update batao",
        "refund chahiye payment ka",
        "payment stuck hai",
        "payment pending hai",
        "transaction issue hai",
        "paise chale gaye sir",
        "payment verify karo",
        "order nahi bana payment ke baad",
    ],

    "speak_to_agent": [

        # ===================== ENGLISH =====================
        "speak to an agent", "connect me to customer care",
        "I want to talk to a person", "get me a human",
        "transfer to support", "speak to the manager",
        "I need customer support", "live agent",
        "talk to a real person", "human agent please",

        "get me customer care executive", "supervisor please",
        "escalate my issue", "I want to escalate",
        "connect me to someone who can help",

        "I don’t want to talk to bot",
        "connect me to human support",
        "let me talk to real person",
        "I need to speak with support team",
        "can I talk to your executive",
        "please transfer my call",
        "I want to speak to someone",
        "this is not helping connect me to agent",
        "I need assistance from human",

        # ===================== HINGLISH =====================
        "agent se baat karni hai", "customer care connect karo",
        "manager chahiye", "insaan se baat karni hai",
        "customer executive chahiye",

        "mujhe agent se baat karni hai",
        "bot se baat nahi karni",
        "real person se connect karo",
        "call transfer karo agent ko",
        "support team se baat karni hai",
        "kisi insaan se connect karo",
        "mujhe help chahiye agent se",
        "manager se baat karni hai",
        "issue escalate karo",

        # ===================== HINDI =====================
        "agent se baat karni hai", "customer care connect karo",
        "manager chahiye", "insaan se baat karni hai",

        "mujhe kisi vyakti se baat karni hai",
        "kripya mujhe agent se jodiye",
        "mujhe sahayata chahiye",
        "is samasya ko escalate karein",
        "mujhe supervisor se baat karni hai",
        "main kisi insaan se baat karna chahta hoon",

        # ===================== MARATHI =====================
        "माणसाशी बोलायचे आहे", "एजंटशी बोलायचे आहे",

        "मला एजंटशी बोलायचे आहे",
        "मला ग्राहक सेवा प्रतिनिधीशी बोलायचे आहे",
        "कॉल ट्रान्सफर करा",
        "मला मॅनेजरशी बोलायचे आहे",
        "ही समस्या escalate करा",
        "मला मदत हवी आहे एजंटकडून",

        # ===================== REAL CALL VARIATIONS =====================
        "agent lagao",
        "call transfer karo",
        "human connect karo",
        "real banda chahiye",
        "kisi se baat karwao",
        "agent chahiye",
        "support se connect karo",
        "call forward karo",
        "ye bot samajh nahi raha",
        "agent se connect karao",
        "customer care se baat karni hai",
    ],

    "other": [

        # ===================== BASIC =====================
        "hello", "hi", "okay", "yes", "no", "thanks", "thank you",
        "bye", "good morning", "good afternoon", "good evening",

        "alright", "sure", "hmm", "uh huh", "right", "of course",
        "one moment", "hold on", "please wait", "I see",
        "okay sir", "got it", "acha", "theek hai", "haan", "nahi",
        "bilkul", "dhanyavaad", "shukriya", "namaste",

        # ===================== CALL CENTER NOISE =====================
        "you have some queries", "regarding",
        "can you speak a little bit", "what do you say",
        "my leader", "e-voice", "yeah thanks", "speaking",
        "remind you", "hello hello", "1 2 3 4",

        "is there anything else I can help you with",
        "how can I help you", "how may I assist you",

        "please hold", "one minute sir", "just a moment",
        "I understand", "noted", "I will note that",

        # ===================== REAL CALL FILLERS =====================
        "hmm ok", "ok ok", "haan bolo",
        "sunai nahi de raha", "hello awaz aa rahi hai",
        "network issue hai", "thoda rukko",
        "ek minute", "ruk jaiye",
        "boliye", "haan ji",

        "testing testing", "check check",
        "awaz aa rahi hai kya",
        "line clear nahi hai",

        # ===================== SHORT NON-INTENT PHRASES =====================
        "yes sir", "no sir", "ok sir",
        "done", "ho gaya", "theek hai sir",
        "samajh gaya", "samajh gaye",
        "thank you sir", "thanks sir",
    ],
}

# ── Train ──────────────────────────────────────────────────────────
print("⏳ Loading sentence encoder (downloads ~117MB first time)...")
encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

texts, labels = [], []
for intent_name, examples in INTENT_EXAMPLES.items():
    for ex in examples:
        texts.append(ex)
        labels.append(intent_name)

print(f"⏳ Encoding {len(texts)} training examples...")
t0 = time.time()
X  = encoder.encode(texts, batch_size=64, show_progress_bar=True)
le = LabelEncoder()
y  = le.fit_transform(labels)

print("⏳ Training LogisticRegression classifier...")
clf = LogisticRegression(max_iter=2000, C=3.0, solver="lbfgs", multi_class="multinomial")
clf.fit(X, y)
print(f"✅ Training complete in {time.time()-t0:.1f}s")

# ── Quick accuracy check ───────────────────────────────────────────
from sklearn.model_selection import cross_val_score
import numpy as np
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print(f"📊 5-fold CV accuracy: {np.mean(scores):.1%} ± {np.std(scores):.1%}")

# ── Save as pickle ─────────────────────────────────────────────────
model_bundle = {
    "encoder":        encoder,
    "classifier":     clf,
    "label_encoder":  le,
    "intent_names":   list(INTENT_EXAMPLES.keys()),
    "num_examples":   len(texts),
    "trained_at":     time.strftime("%Y-%m-%d %H:%M:%S"),
}

with open(SAVE_PATH, "wb") as f:
    pickle.dump(model_bundle, f)

size_mb = os.path.getsize(SAVE_PATH) / 1e6
print(f"\n✅ Model saved to Google Drive:")
print(f"   Path: {SAVE_PATH}")
print(f"   Size: {size_mb:.1f} MB")
print(f"   Intents: {len(INTENT_EXAMPLES)}")
print(f"   Examples: {len(texts)}")
print(f"\n💡 This file is permanently saved in your Google Drive.")
print(f"   Cell 2 (server) loads it automatically — no retraining needed.")

# ── Quick test ─────────────────────────────────────────────────────
print("\n── Quick intent test ─────────────────────────────")
test_phrases = [
    "my vehicle number is MH12AK1234",
    "where is my order",
    "I want to return this brake pad",
    "send me the GST invoice",
    "is clutch plate available",
    "payment failed but money deducted",
    "okay thank you",
    "price of engine oil",
]
for phrase in test_phrases:
    emb  = encoder.encode([phrase], show_progress_bar=False)
    pred = clf.predict(emb)[0]
    prob = clf.predict_proba(emb)[0]
    name = le.inverse_transform([pred])[0]
    conf = max(prob)
    status = "✅" if conf >= 0.6 else "⚠️ "
    print(f"  {status} [{conf:.0%}] {name:<20} ← \"{phrase}\"")

print("\n🎯 If accuracy looks good, run Cell 2 to start the server.")
