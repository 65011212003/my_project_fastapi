# การประมวลผลข้อความสำหรับการวิเคราะห์วรรณกรรมทางวิทยาศาสตร์ 🧬

## ภาพรวมของขั้นตอนการประมวลผลใน `test_preprocessing.py`

เอกสารนี้อธิบายเทคนิคการประมวลผลที่ใช้ในไปป์ไลน์การวิเคราะห์ข้อความทางวิทยาศาสตร์ของเรา สามารถดูโค้ดได้ที่ [`test_preprocessing.py`](test_preprocessing.py)

### 1. การเก็บรักษาข้อความต้นฉบับ 📝
- เก็บข้อความดิบในคอลัมน์ "Original"
- รักษาอักขระพิเศษเช่น `<i>`, `</i>`
- คงเครื่องหมายวรรคตอนและการจัดรูปแบบทั้งหมด
- การนำไปใช้: ฟังก์ชัน `preprocess_text()` เก็บข้อความต้นฉบับใน `steps["original"]`

### 2. การแปลงเป็นตัวพิมพ์เล็ก ⬇️
- แปลงข้อความทั้งหมดเป็นตัวพิมพ์เล็ก
- รักษาเครื่องหมายวรรคตอนและอักขระพิเศษ
- ตัวอย่าง: "Rice" → "rice", "DNA" → "dna"
- การนำไปใช้: `steps["lowercase"] = text.lower()`

### 3. การลบอักขระพิเศษ 🔄
- ลบเครื่องหมายวรรคตอนทั้งหมด
- ลบแท็ก HTML (เช่น `<i>`, `</i>`)
- ลบวงเล็บ (), [], {}
- การนำไปใช้: ใช้ regex `re.sub(r'[^a-zA-Z\s]', '')` ใน `preprocess_text()`

### 4. การแบ่งคำ 🔍
- แยกข้อความเป็นคำแต่ละคำ
- ใช้ `word_tokenize` ของ NLTK
- เก็บคำในรูปแบบลิสต์
- การนำไปใช้: `steps["tokens"] = word_tokenize(steps["no_special_chars"])`

### 5. การลบคำหยุด 🚫
- ลบคำหยุดภาษาอังกฤษโดยใช้ NLTK
- กำจัดคำทั่วไปเช่น "the", "is", "at"
- ลบคำเชื่อมและคำนำหน้านาม
- การนำไปใช้: ใช้ `nltk.corpus.stopwords` ใน `preprocess_text()`

### 6. การลดรูปคำ 🌱
- แปลงคำเป็นรูปพื้นฐานโดยใช้ `WordNetLemmatizer` ของ NLTK
- ตัวอย่าง:
  - "studies" → "study"
  - "leaves" → "leaf"
  - "genes" → "gene"
- การนำไปใช้: `lemmatizer.lemmatize()` ใน `preprocess_text()`

### 7. การลบคำสั้น 📏
- ลบคำที่สั้นกว่า 3 ตัวอักษร
- กรองคำย่อบางคำออก
- การนำไปใช้: `[token for token in steps["lemmatized"] if len(token) > 2]`

### 8. การประมวลผลขั้นสุดท้าย ✨
- รวมขั้นตอนการประมวลผลทั้งหมด
- ลบช่องว่างที่ไม่จำเป็น
- สร้างข้อความต่อเนื่องโดยไม่มีตัวคั่น
- การนำไปใช้: `steps["final"] = ' '.join(steps["no_short_words"])`

## ประโยชน์หลัก 🎯
1. ลดความซ้ำซ้อนของข้อมูล
2. เพิ่มประสิทธิภาพการวิเคราะห์
3. ทำให้รูปแบบข้อมูลเป็นมาตรฐาน
4. อำนวยความสะดวกในการประมวลผลต่อไป
5. ลดความต้องการพื้นที่จัดเก็บ
6. เพิ่มความแม่นยำในการวิเคราะห์

## ข้อควรพิจารณาที่สำคัญ ⚠️
- อาจสูญเสียข้อมูลเชิงความหมายบางส่วน
- ขั้นตอนการประมวลผลควรเหมาะสมกับงาน
- ควรเก็บรักษาข้อมูลต้นฉบับเสมอ
- อาจไม่จำเป็นต้องใช้ทุกขั้นตอน
- ต้องตรวจสอบผลลัพธ์อย่างสม่ำเสมอ

## การบูรณาการโค้ด 🔗
ไปป์ไลน์การประมวลผลนี้ถูกนำไปใช้ใน:
- [`preprocessing.py`](preprocessing.py) - ฟังก์ชันการประมวลผลหลัก
- [`test_preprocessing.py`](test_preprocessing.py) - การทดสอบและสาธิต
- ใช้ไลบรารี: NLTK, spaCy, BioPython

