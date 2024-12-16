#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"
#include "CameraWebServer_AP.h"

WiFiServer server(100);
WebServer webserver(80);

#define RXD2 33
#define TXD2 4
CameraWebServer_AP CameraWebServerAP;

bool WA_en = false;

#define FORWARD 1
#define BACKWARD 2
#define LEFT 3
#define RIGHT 4
#define STOP 0

void ApplicationFunctionSet_SmartRobotCarMotionControl(int direction, int speed) {
    switch (direction) {
        case FORWARD:
            // Code to move the car forward
            Serial.println("Moving Forward");
            break;
        case BACKWARD:
            // Code to move the car backward
            Serial.println("Moving Backward");
            break;
        case LEFT:
            // Code to turn the car left
            Serial.println("Turning Left");
            break;
        case RIGHT:
            // Code to turn the car right
            Serial.println("Turning Right");
            break;
        case STOP:
            // Code to stop the car
            Serial.println("Stopping");
            break;
        default:
            Serial.println("Unknown direction");
            break;
    }
}

void SocketServer_Test(void)
{
  static bool ED_client = true;
  WiFiClient client = server.available(); 
  if (client)                            
  {
    WA_en = true;
    ED_client = true;
    Serial.println("[Client connected]");
    String readBuff;
    String sendBuff;
    uint8_t Heartbeat_count = 0;
    bool Heartbeat_status = false;
    bool data_begin = true;
    while (client.connected()) 
    {
      if (client.available()) 
      {
        char c = client.read();             
        Serial.print(c);                    
        if (true == data_begin && c == '{') 
        {
          data_begin = false;
        }
        if (false == data_begin && c != ' ')
        {
          readBuff += c;
        }
        if (false == data_begin && c == '}')
        {
          data_begin = true;
          if (true == readBuff.equals("{Heartbeat}"))
          {
            Heartbeat_status = true;
          }
          else
          {
            Serial2.print(readBuff);
          }
          readBuff = "";
        }
      }
      if (Serial2.available())
      {
        char c = Serial2.read();
        sendBuff += c;
        if (c == '}')
        {
          client.print(sendBuff);
          Serial.print(sendBuff);
          sendBuff = "";
        }
      }

      static unsigned long Test_time = 0;
      if (millis() - Test_time > 1000)
      {
        Test_time = millis();
        if (0 == (WiFi.softAPgetStationNum()))
        {
          Serial2.print("{\"N\":100}");
          break;
        }
      }
    }
    Serial2.print("{\"N\":100}");
    client.stop();
    Serial.println("[Client disconnected]");
  }
  else
  {
    if (ED_client == true)
    {
      ED_client = false;
      Serial2.print("{\"N\":100}");
    }
  }
}

void FactoryTest(void)
{
  static String readBuff;
  String sendBuff;
  if (Serial2.available())
  {
    char c = Serial2.read();
    readBuff += c;
    if (c == '}')
    {
      if (true == readBuff.equals("{BT_detection}"))
      {
        Serial2.print("{BT_OK}");
        Serial.println("Factory...");
      }
      else if (true == readBuff.equals("{WA_detection}"))
      {
        Serial2.print("{");
        Serial2.print(CameraWebServerAP.wifi_name);
        Serial2.print("}");
        Serial.println("Factory...");
      }
      readBuff = "";
    }
  }
  {
    if ((WiFi.softAPgetStationNum()))
    {
      if (true == WA_en)
      {
        digitalWrite(13, LOW);
        Serial2.print("{WA_OK}");
        WA_en = false;
      }
    }
    else
    {
      static unsigned long Test_time;
      static bool en = true;
      if (millis() - Test_time > 100)
      {
        if (false == WA_en)
        {
          Serial2.print("{WA_NO}");
          WA_en = true;
        }
        if (en == true)
        {
          en = false;
          digitalWrite(13, HIGH);
        }
        else
        {
          en = true;
          digitalWrite(13, LOW);
        }
        Test_time = millis();
      }
    }
  }
}

void handleControl() {
    String direction = webserver.arg("direction");
    Serial.println("Direction received: " + direction);
    
    // Send the command to the Arduino via Serial2
    if (direction == "FORWARD") {
        Serial2.println("FORWARD");
    } else if (direction == "BACKWARD") {
        Serial2.println("BACKWARD");
    } else if (direction == "LEFT") {
        Serial2.println("LEFT");
    } else if (direction == "RIGHT") {
        Serial2.println("RIGHT");
    } else if (direction == "STOP") {
        Serial2.println("STOP");
    } else if (direction == "AUTONOMOUS") {
        Serial2.println("AUTONOMOUS");
    }

    
    // Add CORS headers
    webserver.sendHeader("Access-Control-Allow-Origin", "*");
    webserver.sendHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    webserver.sendHeader("Access-Control-Allow-Headers", "Content-Type");
    
    webserver.send(200, "text/plain", "Command received: " + direction);
}

void setup() {
  // Increase the baud rate from 9600 to 115200
  Serial.begin(115200);
  Serial2.begin(115200, SERIAL_8N1, RXD2, TXD2);
  //http://192.168.4.1/control?var=framesize&val=3
  //http://192.168.4.1/Test?var=
  CameraWebServerAP.CameraWebServer_AP_Init();
  server.begin();
  delay(100);
  pinMode(13, OUTPUT);
  digitalWrite(13, HIGH);
  Serial.println("Elegoo-2020...");
  Serial2.print("{Factory}");

  //Web Server
  webserver.on("/control", HTTP_GET, handleControl); // Handle incoming control commands
  webserver.begin(); // Start the server
  Serial.println("Web server started");
}

void loop() {
  webserver.handleClient(); // Allow the web server to process incoming requests
  SocketServer_Test();
  FactoryTest();
}
