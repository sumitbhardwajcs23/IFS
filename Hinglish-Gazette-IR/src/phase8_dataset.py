"""Phase 8: Build the 200-query Hinglish evaluation benchmark grounded in the Gazette corpus."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config import CHUNKS_JSONL_PATH, EVAL_QUERIES_PATH
from src.storage import read_jsonl, write_jsonl


@dataclass
class TopicSpec:
    topic_id: str
    description: str
    required_patterns: list[str]
    optional_doc_prefix: str | None
    queries: dict[str, str]  # keys: normal_hinglish, phonetic_typo, dialect_regional, informal_question


# 50 grounded UP Gazette information needs x 4 query styles = 200 evaluation queries
TOPIC_SPECS: list[TopicSpec] = [
    TopicSpec(
        topic_id="T01",
        description="Ayurveda medical officer provisional appointment & certificate verification",
        required_patterns=[r"आयुर्वेद\s*चिकित्साधिकार", r"औपबन्धिक", r"प्रमाण"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Ayurved medical officer provisional appointment aur certificate rules",
            "phonetic_typo": "Ayurvedik chikitsa adhikari niyukti praman patra asatya",
            "dialect_regional": "Sarkari hakim doctor bharti mein kagazat janch ki shart",
            "informal_question": "Ayurvedic doctor ki naukri mein galat praman patra dene par kya hoga?",
        },
    ),
    TopicSpec(
        topic_id="T02",
        description="Ayurveda officer probation period and transfer restriction",
        required_patterns=[r"परिवीक्षा\s*अवधि", r"स्थानान्तरण", r"चिकित्सकों"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Probation period mein medical officer transfer rules",
            "phonetic_typo": "Pariviksha avadhi me chikitsak sthanantaran anumanya",
            "dialect_regional": "Naye bharti doctor ka badli tabadla kab tak nahi hoga",
            "informal_question": "Kya probation time mein ayurved doctor transfer le sakte hain?",
        },
    ),
    TopicSpec(
        topic_id="T03",
        description="Ayurveda medical officer affidavit of no criminal prosecution",
        required_patterns=[r"अभियोजन", r"शपथ-पत्र", r"चिकित्सा\s*परिषद"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Ayurveda medical officer joining shapath patra aur court case affidavit",
            "phonetic_typo": "Abhiyojan nyayalay shapat patra chikitsa parishad",
            "dialect_regional": "Doctor naukri join karne se pehle kachahari ka halafnama",
            "informal_question": "Bharti ke time medical council registration aur shapath patra kyun chahiye?",
        },
    ),
    TopicSpec(
        topic_id="T04",
        description="Provincial Development Service promotion to Pay Matrix Level-11",
        required_patterns=[r"प्रादेशिक\s*विकास\s*सेवा", r"पदोन्नति", r"पे-मैट्रिक्स\s*लेवल-11"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Pradeshik vikas seva officer promotion pay matrix level 11",
            "phonetic_typo": "Pradesik vikas sewa padonati pay metrix level 11 vetanman",
            "dialect_regional": "BDO sahab ki tarakki aur tankhwa level 11 ka shasanadesh",
            "informal_question": "Khand vikas adhikari ko promotion ke baad kitni salary milegi?",
        },
    ),
    TopicSpec(
        topic_id="T05",
        description="High Court Registrar-cum-Principal Private Secretary Level 13-A",
        required_patterns=[r"Registrar-cum-Principal Private Secretary", r"Level 13-A"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "High Court Allahabad Registrar Principal Private Secretary pay matrix level 13A",
            "phonetic_typo": "Allahabad highcort registrar sekretary pay metrix 13A",
            "dialect_regional": "Uchch nyayalaya Prayagraj afsar vetan level 13A aadesh",
            "informal_question": "Allahabad High Court ke registrar cadre ko level 13-A kaise mila?",
        },
    ),
    TopicSpec(
        topic_id="T06",
        description="Fast Track Court judges for crimes against women",
        required_patterns=[r"Fast Track Court", r"crimes of crime against women|crime against women"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Fast Track Court civil judge appointment for crime against women",
            "phonetic_typo": "Fast trak cort mahila apradh civil jaj niyukti",
            "dialect_regional": "Aurat aur mahila ke khilaf jurm ke liye alag adalat aur judge",
            "informal_question": "Mahilaon ke khilaf case dekhne ke liye kaun se judge appoint hue?",
        },
    ),
    TopicSpec(
        topic_id="T07",
        description="Judicial Magistrate First Class appointment under BNSS 2023",
        required_patterns=[r"Bharatiya Nagarik Suraksha Sanhita, 2023", r"Judicial Magistrate, First Class"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Judicial Magistrate First Class appointment under Bharatiya Nagarik Suraksha Sanhita 2023",
            "phonetic_typo": "Judishial majistrate first klas BNSS 2023 section 9",
            "dialect_regional": "Naya kanun BNSS ke tahat adalat mein magistrate sahab ki tainati",
            "informal_question": "BNSS 2023 dhara 9 ke andar Judicial Magistrate kaise niyukt hue?",
        },
    ),
    TopicSpec(
        topic_id="T08",
        description="Special Judge Essential Commodities Act 1981",
        required_patterns=[r"Essential Commodities", r"Special Judge"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "UP Essential Commodities Act 1981 Special Judge appointment",
            "phonetic_typo": "Esential comodities akt 1981 speshial jaj section 12A",
            "dialect_regional": "Zaruri saman kanun ke mukadme sunne wale vishesh nyayadhish",
            "informal_question": "Essential Commodities Act 12-A ke liye special judge kaun banaya gaya?",
        },
    ),
    TopicSpec(
        topic_id="T09",
        description="Gram Nyayalaya Nyayadhikari transfers",
        required_patterns=[r"Nyayadhikari, Gram Nyayalaya"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Gram Nyayalaya Nyayadhikari transfer and posting notification",
            "phonetic_typo": "Gram nyayalay nyayadhikari posting transfur order",
            "dialect_regional": "Gaon ki adalat mein nyayadhikari sahab ki badli",
            "informal_question": "Gram Nyayalaya ke judge ka transfer kahan kahan hua hai?",
        },
    ),
    TopicSpec(
        topic_id="T10",
        description="Chief Judicial Magistrate Ghaziabad posting",
        required_patterns=[r"Chief Judicial Magistrate, Ghaziabad"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Chief Judicial Magistrate Ghaziabad appointment order",
            "phonetic_typo": "Chif judishial majistrate Gaziabad notification",
            "dialect_regional": "Ghaziabad zila adalat ke mukhya magistrate ki niyukti",
            "informal_question": "Ghaziabad ke naye CJM kaun banaye gaye hain?",
        },
    ),
    TopicSpec(
        topic_id="T11",
        description="UP Consolidation of Holdings Act 1953 Section 6(1)",
        required_patterns=[r"जोत\s*चकबन्दी\s*अधिनियम", r"धारा-?6"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "UP Jot Chakbandi Adhiniyam 1953 Section 6 notification",
            "phonetic_typo": "Uttar pradesh chakbandi adhiniyam 1953 dhara 6 up-dhara 1",
            "dialect_regional": "Gaon ke khet aur jameen ki chakbandi band karne ka kanun",
            "informal_question": "Chakbandi act ki dhara 6(1) ke tahat kaun si adhisochna nikli hai?",
        },
    ),
    TopicSpec(
        topic_id="T12",
        description="Land acquisition for Vyomkesh Residential Scheme in Rampur",
        required_patterns=[r"Vyomkesh Residential Scheme", r"Rampur"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Rampur Vyomkesh Residential Scheme land acquisition notification",
            "phonetic_typo": "Rampur Vyomkesh residenshal skim land akuisition",
            "dialect_regional": "Rampur jila mein awasiya yojana ke liye jameen adhigrahan",
            "informal_question": "Rampur ki Vyomkesh scheme ke liye kaun si bhumi li ja rahi hai?",
        },
    ),
    TopicSpec(
        topic_id="T13",
        description="Gorakhpur Development Authority Kalesar industrial land acquisition",
        required_patterns=[r"Development Authority, Gorakhpur", r"Kalesar"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Gorakhpur Development Authority Kalesar village land acquisition",
            "phonetic_typo": "Gorakhpur devlopment athority Kalesar indastrial projekt",
            "dialect_regional": "Gorakhpur Kalesar gaon mein factory project ke liye jameen arjan",
            "informal_question": "Gorakhpur Kalesar industrial project ke liye bhumi adhigrahan kab hua?",
        },
    ),
    TopicSpec(
        topic_id="T14",
        description="Sambhal Chandausi village land rehabilitation and resettlement",
        required_patterns=[r"सम्भल", r"चन्दौसी", r"पुनर्वासन"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Sambhal Chandausi Fatehpur Sharifnagar land rehabilitation notification",
            "phonetic_typo": "Sambhal Chandaosi bhumi punarvasan punarvyavasthapan",
            "dialect_regional": "Chandausi gaon ke visthapit parivaron ki jameen aur basawat",
            "informal_question": "Sambhal ke Chandausi mein visthapit logon ke punarvasan ke liye kitni bhumi hai?",
        },
    ),
    TopicSpec(
        topic_id="T15",
        description="Gautam Buddh Nagar Dadri Gram Sabha land exchange",
        required_patterns=[r"गौतमबुद्धनगर", r"दादरी", r"विनिमय"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Gautam Buddh Nagar Dadri Rathi Industries land exchange order",
            "phonetic_typo": "Gautambudh nagar Dadri bhumi vinimay shreni parivartan",
            "dialect_regional": "Dadri tehsil gram sabha ki jameen badalne ka shasanadesh",
            "informal_question": "Gautam Buddh Nagar Dadri mein gram sabha bhumi vinimay kaise hua?",
        },
    ),
    TopicSpec(
        topic_id="T16",
        description="Ghaziabad SDM land exchange circle rate valuation",
        required_patterns=[r"गाजियाबाद", r"विनिमय", r"उप\s*जिलाधिकारी"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Ghaziabad SDM land exchange and category change rules",
            "phonetic_typo": "Gaziabad up jiladhikari bhumi vinimay shreni parivartan",
            "dialect_regional": "Ghaziabad tehsil mein sarkar se jameen adla badli ki shart",
            "informal_question": "Ghaziabad mein bhumi vinimay par SDM ki kya report zaruri hai?",
        },
    ),
    TopicSpec(
        topic_id="T17",
        description="Lucknow Sarojini Nagar land revenue area table",
        required_patterns=[r"लखनऊ\s*सरोजनी\s*नगर"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Lucknow Sarojini Nagar land area hectare revenue table",
            "phonetic_typo": "Lakhnau Sarojni nagar hekteyar bhumi rajaswa",
            "dialect_regional": "Lucknow Sarojini Nagar tehsil ke khet aur rakba ki suchi",
            "informal_question": "Sarojini Nagar Lucknow ki kaun si bhumi ka naksha aur area gazette mein hai?",
        },
    ),
    TopicSpec(
        topic_id="T18",
        description="DLRC land resumption under 1974 Government Order",
        required_patterns=[r"डी0एल0आर0सी0", r"पुनर्ग्रहण"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "DLRC land resumption government order rajaswa notification",
            "phonetic_typo": "DLRC punargrahan shasanadesh rajaswa 1974",
            "dialect_regional": "Rajaswa vibhag dwara sarkari jameen wapas lene ka hukm",
            "informal_question": "DLRC punargrahan aadesh kis purane shasanadesh ke hisab se hai?",
        },
    ),
    TopicSpec(
        topic_id="T19",
        description="Lucknow Nagar Nigam Fecal Sludge and Septage Management MoU",
        required_patterns=[r"फीकल\s*स्लज", r"होस्ट\s*यूएलबी", r"क्लस्टर\s*यूएलबी"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Lucknow Nagar Nigam fecal sludge septage management host ULB MoU",
            "phonetic_typo": "Lakhnau nagar nigam fikal slaj septej host ULB klaster",
            "dialect_regional": "Shehar aur gram panchayat ki gandi nali tanki safai ka samjhauta",
            "informal_question": "Host ULB aur cluster ULB ke beech septage treatment MoU ke kya niyam hain?",
        },
    ),
    TopicSpec(
        topic_id="T20",
        description="15 days prior notice under ULB septage MoU",
        required_patterns=[r"15\s*दिन\s*पहले", r"फीकल\s*स्लज"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Host ULB 15 days notice rule for fecal sludge disposal",
            "phonetic_typo": "Host ULB 15 din pahle suchna fikal slaj nistaran",
            "dialect_regional": "Safai plant mein kachra na lene par 15 din pehle khabar dena",
            "informal_question": "Agar host ULB septage treat na kar paye toh kitne din pehle notice dega?",
        },
    ),
    TopicSpec(
        topic_id="T21",
        description="Zila Panchayat cattle fair and market inspection rules",
        required_patterns=[r"पशु\s*मेला", r"पशु\s*बाज़ार", r"जिला\s*पंचायत"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Zila Panchayat pashu mela aur pashu bazar inspection rules",
            "phonetic_typo": "Jila panchayat pashu mela bajar paith nirikshan safai",
            "dialect_regional": "Gaon dehat mein jaanwar haat aur pashu mela lagane ke niyam",
            "informal_question": "Pashu mela aur pashu bazar ka nirikshan kaun se adhikari karte hain?",
        },
    ),
    TopicSpec(
        topic_id="T22",
        description="Gonda Zila Panchayat bylaw amendment",
        required_patterns=[r"जिलाधिकारी\s*गोण्डा", r"उपविधि"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Gonda Zila Panchayat upvidhi amendment notification",
            "phonetic_typo": "Gonda jiladhikari kshetra panchayat upvidhi sanshodhan",
            "dialect_regional": "Gonda jila panchayat ke kanun mein badlav ka rajpatra",
            "informal_question": "Gonda district panchayat bylaws mein kya amendment hua hai?",
        },
    ),
    TopicSpec(
        topic_id="T23",
        description="Building plan map fee and 25 percent stationery deduction",
        required_patterns=[r"भवन\s*मानचित्र", r"25\s*प्रतिशत", r"स्टेशनरी\s*शुल्क"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Building map approval fee and 25 percent deduction on rejection",
            "phonetic_typo": "Bhavan manchitra shulk raseed 25 pratishat steshnari katoti",
            "dialect_regional": "Ghar ka naksha pass na hone par jama paisa kitna wapas milega",
            "informal_question": "Agar makan ka naksha aswikrit ho jaye toh kitna fees kata jayega?",
        },
    ),
    TopicSpec(
        topic_id="T24",
        description="Nagar Panchayat Banda advertisement banner poster bylaws",
        required_patterns=[r"बण्डा", r"विज्ञापन", r"बैनर|पोस्टर"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Nagar Panchayat Banda advertisement banner poster rules",
            "phonetic_typo": "Nagar panchayat Banda vigyapan bainar postar upvidhi",
            "dialect_regional": "Banda shahar mein diwar aur sadak par poster lagane ka niyam",
            "informal_question": "Nagar Panchayat Banda mein sarkari banner aur hoarding ke kya rules hain?",
        },
    ),
    TopicSpec(
        topic_id="T25",
        description="Contract cancellation and security deposit forfeiture",
        required_patterns=[r"ठेका\s*निरस्त", r"जमानत"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Contract cancellation and security deposit forfeiture penalty rules",
            "phonetic_typo": "Theka nirast jamanat dhanrashi jabt aniyamita",
            "dialect_regional": "Galti karne par thekedar ka theka radd aur jamanat jabti",
            "informal_question": "Serious irregularity par theka cancel hone se jamanat rashi ka kya hota hai?",
        },
    ),
    TopicSpec(
        topic_id="T26",
        description="Daily penalty fine of Rs 25 for bylaw violation",
        required_patterns=[r"25\s*\(?पच्चीस", r"अर्थदण्ड"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Daily fine penalty Rs 25 rupees per day bylaw violation",
            "phonetic_typo": "25 rupaye pratidin arthdand jurmana niyam",
            "dialect_regional": "Niyam todne par rozana pachis rupaya jurmana ka hukm",
            "informal_question": "Kis niyam ko todne par roz 25 rupaye ka arthdand lagta hai?",
        },
    ),
    TopicSpec(
        topic_id="T27",
        description="Nagar Panchayat Shergarh license bylaws 2026-27",
        required_patterns=[r"शेरगढ़", r"उपविधि", r"लाइसेन्स"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Nagar Panchayat Shergarh license bylaws 2026-27",
            "phonetic_typo": "Nagar panchayat Shergadh laisens upvidhi 2026 27",
            "dialect_regional": "Shergarh qasba mein dukan aur vyapar license ka niyam",
            "informal_question": "Shergarh Nagar Panchayat ne 2026-27 ke liye kaun si license upvidhi pass ki?",
        },
    ),
    TopicSpec(
        topic_id="T28",
        description="Nagar Panchayat Daranagar Kadadham Kaushambi notification",
        required_patterns=[r"दारानगर\s*कड़ाधाम"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Nagar Panchayat Daranagar Kadadham Kaushambi gazette order",
            "phonetic_typo": "Nagar panchayat Daranagar Kadadham Kaosambi aadesh",
            "dialect_regional": "Kaushambi Daranagar nagar panchayat ki official vigyapti",
            "informal_question": "Daranagar Kadadham Nagar Panchayat ka kaun sa notification aaya hai?",
        },
    ),
    TopicSpec(
        topic_id="T29",
        description="Property ownership transfer mutation and lease deed rules",
        required_patterns=[r"नामान्तरण", r"पट्टा\s*विलेख"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Property ownership transfer mutation namantaran and lease deed copy",
            "phonetic_typo": "Sampatti hastantaran namantaran patta vilekh pratilipi",
            "dialect_regional": "Ghar ya plot kharidne par naam chadhane aur patta kagaz ke niyam",
            "informal_question": "Property transfer ke baad namantaran karwane ke liye kya documents lagte hain?",
        },
    ),
    TopicSpec(
        topic_id="T30",
        description="Tax assessment list modification and public objection",
        required_patterns=[r"निर्धारण\s*सूची", r"संशोधित"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Property tax assessment list name entry modification and objection",
            "phonetic_typo": "Nirdharan suchi sanshodhit parivartit naam pravisht apatti",
            "dialect_regional": "Nagar palika tax register mein naam darj karne par aapatti",
            "informal_question": "Assessment list mein naam badalwane ke khilaf aapatti kaise lagayi jaati hai?",
        },
    ),
    TopicSpec(
        topic_id="T31",
        description="UP Juvenile Justice Care and Protection Rules 2019 / Rule 87",
        required_patterns=[r"किशोर\s*न्याय|Juvenile Justice", r"87|संरक्षण"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Mahila Kalyan Vibhag Juvenile Justice child protection rule 87 notification",
            "phonetic_typo": "Kishor nyay balak sanrakshan niyamawali mahila kalyan vibhag",
            "dialect_regional": "Bacchon ki dekhbhal aur suraksha ke liye mahila kalyan vibhag ka kanun",
            "informal_question": "Juvenile Justice rules ke rule-87 ke tahat Mahila Kalyan Vibhag ne kya kiya?",
        },
    ),
    TopicSpec(
        topic_id="T32",
        description="State Planning Institute Joint Director DPC promotion",
        required_patterns=[r"राज्य\s*नियोजन\s*संस्थान", r"संयुक्त\s*निदेशक"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Rajya Niyojan Sansthan Joint Director DPC promotion notification",
            "phonetic_typo": "Rajya niyojan sansthan sanyukt nideshak padonati chayan samiti",
            "dialect_regional": "Lucknow planning vibhag mein joint director ki tarakki ka aadesh",
            "informal_question": "Rajya Niyojan Sansthan mein Joint Director promotion committee ka kya order hai?",
        },
    ),
    TopicSpec(
        topic_id="T33",
        description="UP Agriculture Service Group A Deputy Director seniority",
        required_patterns=[r"कृषि\s*सेवा\s*समूह", r"उप\s*कृषि\s*निदेशक"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "UP Agriculture Service Group A Deputy Director seniority order",
            "phonetic_typo": "UP krishi sewa samuh ka up krishi nideshak jyeshthata kramank",
            "dialect_regional": "Kheti vibhag ke bade afsar ki varishtha aur posting order",
            "informal_question": "UP Krishi Sewa Group-A ke Deputy Director ke baare mein kya order nikla?",
        },
    ),
    TopicSpec(
        topic_id="T34",
        description="Finance General Department GPF / Rule 11 notification",
        required_patterns=[r"G-2-249/X-2024", r"FINANCE|नियम"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Finance General Department Rule 11 notification No 6/G-2-249",
            "phonetic_typo": "Fainans jenral dipartment niyam 11 vitt vibhag adhisochna",
            "dialect_regional": "Vitta vibhag ka rule 11 wala sarkari rajpatra aadesh",
            "informal_question": "Finance General Department ne No. 6/G-2-249 mein kis rule ka zikr kiya hai?",
        },
    ),
    TopicSpec(
        topic_id="T35",
        description="Irrigation and Water Resources Department Form 19 Sub-rule 27",
        required_patterns=[r"IRRIGATION AND WATER RESOURCES|Sub-rule \(1\) of rule 27"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Irrigation and Water Resources Department Uttar Pradesh Form 19 rule 27",
            "phonetic_typo": "Irigeshan wotar risorses dipartment UP form 19 sub rul 27",
            "dialect_regional": "Sinchai aur pani vibhag Uttar Pradesh ka Form 19 aur niyam 27",
            "informal_question": "Irrigation department ke Form-19 aur Rule 27(1) mein kya published hua hai?",
        },
    ),
    TopicSpec(
        topic_id="T36",
        description="Collector Balrampur Section 11 Land Acquisition Act notice",
        required_patterns=[r"BALRAMPUR", r"Section 11"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Collector Balrampur Sub-Section 1 Section 11 land acquisition notice",
            "phonetic_typo": "Kalektar Balrampur sekshan 11 bhumi adhigrahan notis",
            "dialect_regional": "Balrampur jila adhikari dwara jameen lene ka kanuni notice",
            "informal_question": "Balrampur Collector ne Section 11 ke tahat kya notification jari kiya?",
        },
    ),
    TopicSpec(
        topic_id="T37",
        description="Newspaper publication notice at Nagar Panchayat expense",
        required_patterns=[r"दो\s*दैनिक\s*समाचार-पत्रों", r"नोटिस"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Two daily newspapers notice publication rule Nagar Panchayat",
            "phonetic_typo": "Do dainik samachar patra notis prakashan nagar panchayat",
            "dialect_regional": "Akhbar mein sarkari notice chhapwane ka kharch aur niyam",
            "informal_question": "Nagar Panchayat ko do daily newspapers mein notice kyun chhapwana padta hai?",
        },
    ),
    TopicSpec(
        topic_id="T38",
        description="Article 348(3) Constitution English translation publication",
        required_patterns=[r"Article-348", r"English translation"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Article 348 clause 3 Constitution English translation gazette order",
            "phonetic_typo": "Artikal 348 kloj 3 konstitushan inglish transleshan",
            "dialect_regional": "Samvidhan ke anuchhed 348 ke tahat angrezi anuvad chhapne ka hukm",
            "informal_question": "Gazette notifications mein Article-348(3) ka zikr kyun hota hai?",
        },
    ),
    TopicSpec(
        topic_id="T39",
        description="Partnership deed Agrotech Solution Gorakhpur",
        required_patterns=[r"एग्रोटेक\s*सॉल्यूशन", r"साझेदारी\s*डीड"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Agrotech Solution Gorakhpur partnership deed gazette notice",
            "phonetic_typo": "Agrotek solyushan Gorakhpur sajhedari did notis",
            "dialect_regional": "Gorakhpur ki company mein hissedari aur partnership ka elan",
            "informal_question": "Messrs Agrotech Solution Gorakhpur ke partnership deed mein kya likha hai?",
        },
    ),
    TopicSpec(
        topic_id="T40",
        description="Name correction in Aadhaar card for Alina to Avani",
        required_patterns=[r"ALINA", r"AVANI"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Gazette name correction Alina to Avani notice",
            "phonetic_typo": "Alina se Avani sahi naam sanshodhan gajet notis",
            "dialect_regional": "Beti ka galat naam sudhar kar naya naam rakhne ki vigyapti",
            "informal_question": "Alina ka naam badalkar Avani karne ka notice kis gazette mein hai?",
        },
    ),
    TopicSpec(
        topic_id="T41",
        description="Name correction in Gazette for Kartike",
        required_patterns=[r"KARTIKE"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Gazette name change correction notice Kartike",
            "phonetic_typo": "Kartike naam galat ankit sahi naam gajet notis",
            "dialect_regional": "Bete ke kagaz mein Kartike naam sudharne ki sarkari suchna",
            "informal_question": "Kartike ke naam correction ka Part 8 notice kahan hai?",
        },
    ),
    TopicSpec(
        topic_id="T42",
        description="Name correction for Mohammad Sameer Shah",
        required_patterns=[r"MOHAMMAD SAMEER SHAH"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Mohammad Sameer Shah name correction gazette Part 8",
            "phonetic_typo": "Mohamad Samir Sah naam sudhar gajet vigyapti",
            "dialect_regional": "Aadhaar aur kagaz mein Mohammad Sameer naam sahi karwane ka notice",
            "informal_question": "Mohammad Sameer Shah ke naam badalne ki suchna kis issue mein chhapi?",
        },
    ),
    TopicSpec(
        topic_id="T43",
        description="Name correction for Tiara Mishthi",
        required_patterns=[r"TIARA MISHTHI"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Tiara Mishthi correct name certificate gazette notice",
            "phonetic_typo": "Tiyara Mishti sahi naam pramanit gajet",
            "dialect_regional": "Putri Tiara Mishthi ke sahi naam ki pahchan wala notice",
            "informal_question": "Tiara Mishthi ka naam correction gazette notification kahan milega?",
        },
    ),
    TopicSpec(
        topic_id="T44",
        description="Ayodhya Bikapur Rajiv Goyal notice",
        required_patterns=[r"RAJIV GOYAL", r"बीकापुर"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Ayodhya Bikapur Rajiv Goyal gazette notice",
            "phonetic_typo": "Ayodhya Bikapur Rajeev Goyal naam gajet",
            "dialect_regional": "Ayodhya tehsil Bikapur ke Rajiv Goyal ka rajpatra notice",
            "informal_question": "Tehsil Bikapur Ayodhya ke Rajiv Goyal ka kaun sa notice chhapa hai?",
        },
    ),
    TopicSpec(
        topic_id="T45",
        description="District Magistrate Rampur judicial/admin order",
        required_patterns=[r"District Magistrate, Rampur"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Office of the District Magistrate Rampur notification",
            "phonetic_typo": "Distrikt Majistrate Rampur ofis aadesh",
            "dialect_regional": "Rampur ke jila adhikari karyalaya ki adhisochna",
            "informal_question": "District Magistrate Rampur office se kaun sa notification aaya hai?",
        },
    ),
    TopicSpec(
        topic_id="T46",
        description="Gorakhpur High Court / Judicial posting or notification",
        required_patterns=[r"Gorakhpur", r"Judge|Magistrate"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Gorakhpur judicial magistrate judge posting notification",
            "phonetic_typo": "Gorakhpur judishial jaj majistrate tainati",
            "dialect_regional": "Gorakhpur adalat mein judge sahab ki badli aur posting",
            "informal_question": "Gorakhpur court mein kis judge ki posting hui hai?",
        },
    ),
    TopicSpec(
        topic_id="T47",
        description="Lucknow Jaliamau Sadar land area records",
        required_patterns=[r"लखनऊ\s*सदर", r"जलियामऊ"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Lucknow Sadar Jaliamau land area revenue notification",
            "phonetic_typo": "Lakhnau sadar Jaliyamau bhumi kshetrafal rajaswa",
            "dialect_regional": "Lucknow Sadar Jaliamau gaon ki jameen aur gata sankhya",
            "informal_question": "Jaliamau Lucknow Sadar ki kaun si zameen gazette mein shamil hai?",
        },
    ),
    TopicSpec(
        topic_id="T48",
        description="Bharatiya Nagrik Suraksha Sanhita Ayodhya judicial appointment",
        required_patterns=[r"Ayodhya", r"Bharatiya Nagarik Suraksha Sanhita"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Ayodhya Judicial Magistrate appointment under BNSS 2023",
            "phonetic_typo": "Ayodhya majistrate niyukti BNSS 2023",
            "dialect_regional": "Ayodhya adalat mein naye magistrate ki tainati ka hukm",
            "informal_question": "Ayodhya mein BNSS 2023 ke tahat kaun Judicial Magistrate bane?",
        },
    ),
    TopicSpec(
        topic_id="T49",
        description="Bulandshahar Civil Judge and Judicial Magistrate posting",
        required_patterns=[r"Bulandshahar", r"Civil Judge|Judicial Magistrate"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Bulandshahar Civil Judge and Judicial Magistrate transfer order",
            "phonetic_typo": "Bulandshahar sivil jaj judishial majistrate badli",
            "dialect_regional": "Bulandshahar kachahari mein judge sahab ki niyukti aur tabadla",
            "informal_question": "Bulandshahar court mein kin judges ka transfer aur posting hua?",
        },
    ),
    TopicSpec(
        topic_id="T50",
        description="UP Gazette annual subscription & part index page",
        required_patterns=[r"सम्पूर्ण\s*गजट\s*का\s*मूल्य", r"वार्षिक"],
        optional_doc_prefix=None,
        queries={
            "normal_hinglish": "Uttar Pradesh Gazette annual subscription price and part index",
            "phonetic_typo": "UP gajet varshik chanda mulya bhag suchi",
            "dialect_regional": "Sarkari rajpatra ki salana keemat aur bhag ki jankari",
            "informal_question": "UP Gazette ka salana chanda aur bhag-1 ka rate kya hai?",
        },
    ),
]


def resolve_topic_qrels(
    topic: TopicSpec,
    chunks: list[dict[str, Any]],
    max_qrels_per_topic: int = 12,
) -> tuple[list[str], list[str]]:
    """Find chunks matching all formal Gazette patterns for `topic`."""
    compiled = [re.compile(pat, re.IGNORECASE) for pat in topic.required_patterns]
    matched_chunks: list[dict[str, Any]] = []

    for chunk in chunks:
        if topic.optional_doc_prefix and not chunk["doc_id"].startswith(topic.optional_doc_prefix):
            continue
        haystack = f"{chunk.get('heading', '')}\n{chunk.get('ministry', '')}\n{chunk['text']}"
        if all(pat.search(haystack) for pat in compiled):
            matched_chunks.append(chunk)

    chunk_ids = [c["chunk_id"] for c in matched_chunks[:max_qrels_per_topic]]
    doc_ids = sorted({c["doc_id"] for c in matched_chunks[:max_qrels_per_topic]})
    return chunk_ids, doc_ids


def build_eval_dataset(
    chunks_path: Path = CHUNKS_JSONL_PATH,
    output_path: Path = EVAL_QUERIES_PATH,
) -> list[dict[str, Any]]:
    """Build the 200-query evaluation benchmark with verified qrels from `chunks.jsonl`."""
    chunks = read_jsonl(chunks_path)
    records: list[dict[str, Any]] = []
    missing_topics: list[str] = []

    q_num = 0
    for topic in TOPIC_SPECS:
        rel_chunks, rel_docs = resolve_topic_qrels(topic, chunks)
        if not rel_chunks:
            missing_topics.append(topic.topic_id)
            continue
        for query_type, query_text in topic.queries.items():
            q_num += 1
            records.append(
                {
                    "query_id": f"Q{q_num:03d}",
                    "topic_id": topic.topic_id,
                    "topic_description": topic.description,
                    "query_type": query_type,
                    "query": query_text,
                    "relevant_chunk_ids": rel_chunks,
                    "relevant_doc_ids": rel_docs,
                }
            )

    if missing_topics:
        raise ValueError(f"Topics had 0 matching chunks in corpus: {missing_topics}")

    write_jsonl(output_path, records)
    return records
