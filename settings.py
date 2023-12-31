VOICE_LANGUAGE = 'ru-RU'

MAX_MESSAGE_SIZE = 1000 * 50  # in bytes
MAX_MESSAGE_DURATION = 15  # in seconds

CLASSESS = ['Assalomu alaykum. Savolingizni aniqroq yozib yuboring iltimos.',
 "Assalomu alaykum. Xozir siz bilan bog'lanib, yordam beramiz.",
 "Assalomu alaykum. Hurmatli mijoz, to'lov tafsilotlarini (To'lov raqami, summasi, vaqti va hokazo) bizga yuboring, tekshirib ko'ramiz.",
 "           Assalomu alaykum. Bizda “To'lovlar tarixi” bo'limi bor. U bo'limda CLICK tizimi orqali amalga oshirilgan to'lovlaringiz tarixi ko'rstailadi. Agar kartangizning barcha kirim va chiqimlarini ko'rmoqchi bo'lsangiz. \n                 1. Ilova  (Click Evolution) \n?? Sozlamalar \nBildirishnomalar\nKartalar bo'yicha tarix \n                 2. Bot  (havola- clickuz)\n?? Tarix\nTo'lovlarni ko'rsatish\n                 3. Sayt  (my.click.uz) \n ?? Tarix \nKarta bo'yicha tarix \n               4. USSD  (*880#)  \n  *880*00*15# - Oxirgi 15 to'lovlar holatini tekshirish \n(Pullik xizmat. Mobil operatoringizga bog'liq)\n   *880*00*3#- Oxirgi 3 to'lovlar holatini tekshirish- bepul\n       Shu bo'lim orqali qilgan to'lovlaringizni ko'rsangiz bo'ladi. Bu xizmat Uzcard kartalari uchun va pullik: oyiga 1000 so'm har bir karta uchun.",
 "Assalomu alaykum. Aynan qanday xatolik bo'lmoqda?",
 "Assalomu alaykum. Agar kartangiz bizning tizimga ulangan bo'lsa, my.click.uz saytimizga kirib, kerakli kartani tanlang, va so'ng “Karta raqamini qo'rsatish” tugmasini bosing. Karta raqami to'liq ko'rsatiladi. Yoki bankingizga murojaat qilishingiz mumkin.",
 "Assalomu alaykum. Mablag'ni yuborgan mijozimiz bizga aynan CLICK tizimiga ulangan raqamlaridan qo'ng'iroq qilishlari lozim. Tel.raqam: +<phone>",
 "Salomat bo'ling.",
 "Assalaamu alaykum. A garda Click Pin kodingizni unutib qo'ygan bo'lsangiz shu bo'limdan to'g'irlab olsangiz bo'ladi ??\n?? Sozlamalar\n1) CLICK-PIN bilan tasdiqlash\n2) CLICK-PIN ni qayta o'rnatish \n??aynan shu bo'limni bossangiz Click Pin kodingizni yangilaysiz va sizga sms tarzda Click ulangan raqamingizga sms kod boradi (bu kodni xech kimga aytmang) so'ng tasdiqlash kodni yuborasiz va yangi Click Pin o'rnatib olasiz. \nEslatib o'tamiz plastik kartangiz Click tizimidan uzulib qolmaydi.",
 "Assalomu alaykum. Bizga hamyoni foallashtirish uchun quyidagi ma'lumotlarni yuboring:\n1. Click-hamyonning faol emasligi aks etgan skrinshot;\n2. Click tizimiga ulangan telefon raqamingiz;\n3. O'zingizning tug'ilgan sanangiz (pasportda ko'rsatilganidek);\n4. Pasport seriya va oxirgi 4 PINFL raqami.",
 'Assalomu alaykum. Qanday yordam bera olaman?',
 "Assalomu alaykum! Click tizimiga ulanish uchun siz birinchi navbatda PLASTIK KARTANGIZGA SMS-XABARNOMAni bankomat orqali eski raqamingizdan ochirasiz, kegin yangi raqamingizga  yoqtirasiz, so'ng Click tizimiga ulaysiz:\n?? Sozlamalar\n?? Hisoblarim\n?? Yangi karta qo'shish",
 "Assalomu alaykum. Hurmatli mijoz, agarda hamyoningiz faol bo'lmasa bizga telegram orqali @Call_centre_click_Official murojaat qiling iltimos. Yordam beramiz.",
 "Assalomu alaykum. Botimizda “To'lovlar tarixi” bo'limi bor. U bo'limda CLICK tizimi orqali amalga oshirilgan to'lovlaringiz tarixi ko'rstailadi. Agar kartangizning barcha kirim va chiqimlarini ko'rmoqchi bo'lsangiz, shu bo'limda “Xizmat sozlamalari” tugmasini bosib, kartani tanlang. Bu xizmat pullik: oyiga 1000 so'm har bir karta uchun.\nUZ Card kartalari uchun bu xizmat.",
 "Assalomu alaykum. Bizga quyidagi ma'lumotlarni yuboring:\n1. Click-hamyonning faol emasligi aks etgan skrinshot;\n2. Click tizimiga ulangan telefon raqamingiz;\n3. O'zingizning tug'ilgan sanangiz (pasportda ko'rsatilganidek).",
 "Assalomu alaykum. To'lov raqamini yozib yuboring, iltimos.",
 'Assalomu alaykum. Afsuski yoq.',
 "Assalomu alaykum. Hurmatli mijoz, to'lovingiz kutish xolatida ekan. Biroz kutishingizni so'raymiz, yani bu to'lovingiz o'tadi yoki qaytadi plastik kartangizga.",
 "Assalomu alaykum. Hurmatli mijoz, tekshirib o'tamiz to'lovingizni, agarda yechilgan bo'lsa ertaga kun davomida plastik kartangizga qaytarib beriladi. Noqulayliklar uchun uzr so'raymiz.",
 "Assalomu alaykum! Hurmatli mijoz, tizimda profilaktika ishlari olib borilayotganligi sababdan vaqtincha to'lov va o'tkazmalarni amalga oshirish imkoni mavjud emas. Tez orada bu profilaktika ishlari o'z nihoyasiga yetadi va to'lovlar ishga tushadi. Keltirilgan noqulayliklar uchun uzur so'raymiz.",
 "Assalomu alaykum. Hurmatli mijoz, e'tiboringiz va taklifingiz uchun katta rahmat. Taklifingizni albatta mutaxasislarimizga etkazib qo'yamiz.",
 "Assalomu alaykum.\nTo'lovlar tarixi\n?? KARTA BO'YICHA TARIX XIZMATI YOQILMAGAN\nFaqat CLICK tizimi orqali amalga oshirilgan to'lovlarni ko'rishingiz mumkin. Xizmatni yoqish/o'chirish uchun Xizmat sozlamalari bo'limiga o'ting:\n?? Xizmat sozlamalari\n?? Karta bo'yicha tarix\n?? Ushbu xizmat Sizning kartangiz uchun sarf-harajatlar va tushumlarning to'liq tarixini ko'rish imkonini beradi (CLICK tizimidan tashqari operatsiyalar bilan birgalikda).\n??XIZMAT HAQI BIR OYGA HAR BIR KARTA UCHUN - 1'000 SO'M\n? - xizmat yoqilgan\n? - xizmat o'chirilgan\nKartani tanlang.",
 "Assalomu alaykum! \nYuqoridagi ma'lumotlar bizga tegishli emas. Xavfsizligingiz uchun amal qilish muddati va  karta raqamini hech kimga aytmaslikni, shuningdek SMS orqali tasdiqlash kodini yubormaslikni maslahat beramiz. Firibgarlardan ehtiyot bo'ling."]
