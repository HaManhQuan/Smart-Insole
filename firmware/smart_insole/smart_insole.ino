/**
 * smart_insole.ino
 * ================
 * Firmware cho Seeed Studio XIAO ESP32C3
 * Đọc 4x FSR402 → truyền BLE 50Hz → PC bác sĩ
 *
 * Phần cứng:
 *   S1 (ụ ngón cái)  → D2 / GPIO4  (ESP32 ADC 12-bit, 0–4095)
 *   S2 (ngón cái)    → ADS1115 ch0 (16-bit, 0–26400 với GAIN_ONE)
 *   S3 (ụ ngón út)   → D0 / GPIO2  (ESP32 ADC 12-bit, 0–4095)
 *   S4 (gót chân)    → D1 / GPIO3  (ESP32 ADC 12-bit, 0–4095)
 *   ADS1115: SDA→D4(GPIO6), SCL→D5(GPIO7), VDD→3.3V, ADDR→GND (0x48)
 *
 * BLE Packet (8 bytes, big-endian uint16):
 *   [0,1] S1 — ESP32 ADC GPIO4  (0–4095)
 *   [2,3] S2 — ADS1115 ch0      (0–26400)
 *   [4,5] S3 — ESP32 ADC GPIO2  (0–4095)
 *   [6,7] S4 — ESP32 ADC GPIO3  (0–4095)
 *
 * Thư viện cần cài:
 *   - NimBLE-Arduino  (by h2zero)      ← thay ArduinoBLE
 *   - Adafruit ADS1X15 (by Adafruit)
 *   - Wire (built-in)
 */

#include <NimBLEDevice.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>

// ---------------------------------------------------------------------------
// Cấu hình chân
// ---------------------------------------------------------------------------

#define PIN_S1   4    // GPIO4 = D2 — ụ ngón cái (ESP32 ADC)
#define PIN_S3   2    // GPIO2 = D0 — ụ ngón út  (ESP32 ADC)
#define PIN_S4   3    // GPIO3 = D1 — gót chân    (ESP32 ADC)
// S2 đọc qua ADS1115 channel 0

#define LED_PIN  10   // XIAO ESP32C3: LED vàng GPIO10

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

#define SAMPLE_INTERVAL_MS   20    // 50Hz = mỗi 20ms 1 sample
#define OVERSAMPLE_COUNT      4    // oversampling để giảm noise ADC

// ---------------------------------------------------------------------------
// BLE UUIDs — phải khớp với bleServices.js
// ---------------------------------------------------------------------------

#define BLE_DEVICE_NAME         "SmartInsole"
#define BLE_SERVICE_UUID        "12345678-1234-5678-1234-56789abcdef0"
#define BLE_CHARACTERISTIC_UUID "12345678-1234-5678-1234-56789abcdef1"

// ---------------------------------------------------------------------------
// Đối tượng toàn cục
// ---------------------------------------------------------------------------

Adafruit_ADS1115 ads;
bool adsReady = false;
NimBLEServer*         pServer   = nullptr;
NimBLECharacteristic* pSensorChar = nullptr;
bool bleConnected = false;

unsigned long lastSampleTime = 0;

// ---------------------------------------------------------------------------
// BLE Server Callbacks — theo dõi connect/disconnect
// ---------------------------------------------------------------------------

class ServerCallbacks : public NimBLEServerCallbacks {
  void onConnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo) override {
    bleConnected = true;
    digitalWrite(LED_PIN, HIGH);
    Serial.println("[BLE] Client kết nối");
  }
  void onDisconnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo, int reason) override {
    bleConnected = false;
    digitalWrite(LED_PIN, LOW);
    Serial.println("[BLE] Client ngắt kết nối — quảng bá lại...");
    NimBLEDevice::startAdvertising();
  }
};

// ---------------------------------------------------------------------------
// Helper: đọc ESP32 ADC với oversampling
// ---------------------------------------------------------------------------

uint16_t readADCOversampled(uint8_t pin) {
  uint32_t sum = 0;
  for (uint8_t i = 0; i < OVERSAMPLE_COUNT; i++) {
    sum += analogRead(pin);
  }
  uint16_t raw = (uint16_t)(sum / OVERSAMPLE_COUNT);
  return 4095 - raw;
}

// ---------------------------------------------------------------------------
// Helper: đọc ADS1115 channel 0 (S2)
// ---------------------------------------------------------------------------

uint16_t readADS1115() {
  if (!adsReady) return 0;
  int16_t raw = ads.readADC_SingleEnded(0);
  if (raw < 0) raw = 0;
  uint16_t val = (uint16_t)raw;
  return (val > 26400) ? 0 : (26400 - val);
}

// ---------------------------------------------------------------------------
// Helper: đóng gói 8 bytes big-endian
// [0,1]=S1  [2,3]=S2  [4,5]=S3  [6,7]=S4
// ---------------------------------------------------------------------------

void packPacket(uint8_t* buf,
                uint16_t s1, uint16_t s2,
                uint16_t s3, uint16_t s4) {
  buf[0] = (s1 >> 8) & 0xFF;  buf[1] = s1 & 0xFF;
  buf[2] = (s2 >> 8) & 0xFF;  buf[3] = s2 & 0xFF;
  buf[4] = (s3 >> 8) & 0xFF;  buf[5] = s3 & 0xFF;
  buf[6] = (s4 >> 8) & 0xFF;  buf[7] = s4 & 0xFF;
}

// ---------------------------------------------------------------------------
// Helper: in Serial debug ở 5Hz (không làm nghẽn vòng lặp 50Hz)
// ---------------------------------------------------------------------------

void printSerial(uint16_t s1, uint16_t s2,
                 uint16_t s3, uint16_t s4,
                 unsigned long ts) {
  static unsigned long lastPrint = 0;
  if (ts - lastPrint < 200) return;
  lastPrint = ts;
  Serial.print("S1="); Serial.print(s1);
  Serial.print(" S2="); Serial.print(s2);
  Serial.print(" S3="); Serial.print(s3);
  Serial.print(" S4="); Serial.print(s4);
  Serial.print(" t="); Serial.println(ts);
}

// ---------------------------------------------------------------------------
// setup()
// ---------------------------------------------------------------------------

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("[SmartInsole] Khởi động...");

  // LED
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // ADC ESP32
  analogReadResolution(12);
  pinMode(PIN_S1, INPUT);
  pinMode(PIN_S3, INPUT);
  pinMode(PIN_S4, INPUT);
  analogSetPinAttenuation(PIN_S1, ADC_11db);
  analogSetPinAttenuation(PIN_S3, ADC_11db);
  analogSetPinAttenuation(PIN_S4, ADC_11db);


  // ADS1115
  Wire.begin(6, 7);   // SDA=GPIO6(D4), SCL=GPIO7(D5)
  adsReady = ads.begin(0x48);
  if (!adsReady) {
    Serial.println("[ERROR] Không tìm thấy ADS1115 tại 0x48!");
    Serial.println("  Kiểm tra: SDA→D4, SCL→D5, VDD→3.3V, ADDR→GND");
  } else {
    ads.setGain(GAIN_ONE);               // ±4.096V, 3.3V → max ~26400
    ads.setDataRate(RATE_ADS1115_250SPS);
    Serial.println("[OK] ADS1115 sẵn sàng (GAIN_ONE)");
  }

  // NimBLE
  NimBLEDevice::init(BLE_DEVICE_NAME);
  // NimBLEDevice::setDeviceName(BLE_DEVICE_NAME);
  NimBLEDevice::setPower(ESP_PWR_LVL_P9);   // công suất tối đa

  pServer = NimBLEDevice::createServer();
  pServer->setCallbacks(new ServerCallbacks());

  NimBLEService* pService = pServer->createService(BLE_SERVICE_UUID);

  pSensorChar = pService->createCharacteristic(
    BLE_CHARACTERISTIC_UUID,
    NIMBLE_PROPERTY::NOTIFY
  );

  pService->start();

  NimBLEAdvertising* pAdv = NimBLEDevice::getAdvertising();
  pAdv->setName(BLE_DEVICE_NAME);
  pAdv->addServiceUUID(BLE_SERVICE_UUID);
  pAdv->start();

  Serial.print("[BLE] Đang quảng bá: ");
  Serial.println(BLE_DEVICE_NAME);
  Serial.println("[BLE] Chờ kết nối từ trình duyệt...");
  Serial.println("--------------------------------------");
}

// ---------------------------------------------------------------------------
// loop()
// ---------------------------------------------------------------------------

void loop() {
  if (!bleConnected) {
    delay(10);
    return;
  }

  unsigned long now = millis();
  if (now - lastSampleTime < SAMPLE_INTERVAL_MS) return;
  lastSampleTime += SAMPLE_INTERVAL_MS;  // ← bù trừ drift, không reset về now
  // Nếu bị tụt lại quá nhiều (ví dụ I2C treo 100ms), reset về now để tránh burst
  if (now - lastSampleTime > SAMPLE_INTERVAL_MS * 3) lastSampleTime = now;

  // Đọc 4 sensor
  uint16_t raw_s1 = readADCOversampled(PIN_S1);
  uint16_t raw_s2 = readADS1115();
  uint16_t raw_s3 = readADCOversampled(PIN_S3);
  uint16_t raw_s4 = readADCOversampled(PIN_S4);

  // Đóng gói + gửi BLE notify
  uint8_t packet[8];
  packPacket(packet, raw_s1, raw_s2, raw_s3, raw_s4);
  pSensorChar->setValue(packet, 8);
  pSensorChar->notify();

  // Debug Serial
  printSerial(raw_s1, raw_s2, raw_s3, raw_s4, now);
}
