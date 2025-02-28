#include <WiFi.h>
#include <Ticker.h> // Include Ticker library

const char* ssid = "khouloud";
const char* password = "koukou123";
int i = 0 ;
const char* host = "192.168.137.9";
const uint16_t port = 5000; // Ensure this port matches the one used by the server on Raspberry Pi
const int sampleRate = 1000; // Sampling rate in Hz
volatile bool sampleFlag = false; // Flag to indicate sampling time
const int emgPin = 34; // Choose the ESP32 pin to read the EMG data

WiFiClient client;
Ticker sampler;

void setup() {
  
  Serial.begin(115200);
  delay(10);

  // Connect to Wi-Fi network
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  // Connect to the server
  Serial.print("Connecting to ");
  Serial.print(host);
  Serial.print(':');
  Serial.println(port);

  if (!client.connect(host, port)) {
    Serial.println("Connection failed.");
    return;
  }

  Serial.println("Connected to server");
  // Attach the sampling function to the Ticker
  sampler.attach_ms(1000 / sampleRate, setSampleFlag);
}

void loop() {
  if (sampleFlag) {
    sampleFlag = false;
    sendEMGData();
  }


}

void setSampleFlag() {
  sampleFlag = true; // Set the flag to indicate it's time to sample
}

void sendEMGData() {
  // Sample and send EMG data to the client
  int emgValue = analogRead(emgPin);
  int emgData = int(emgValue);

  // Send EMG data to the client with a newline character
  if (client.connected()) {
    client.println(emgData);
      //i = i + 1 ;
  }

  Serial.println(i);
}
