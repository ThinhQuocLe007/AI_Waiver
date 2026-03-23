# AI Waiter: System Architecture Documentation

This document outlines the multi-layered architecture of the AI Waiter project, covering the high-level system components, the internal AI Orchestration layers, and the ROS 2 hardware integration.

---

## 1. High-Level System Architecture
This diagram shows the interaction between the Customer, the Robot (Jetson Orin), the Central Management System, and the Kitchen.

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart LR
    %% Styles
    classDef customer fill:#0d47a1,stroke:#64b5f6,stroke-width:2px,color:#fff
    classDef robot fill:#e65100,stroke:#ffb74d,stroke-width:2px,color:#fff
    classDef central fill:#4a148c,stroke:#ce93d8,stroke-width:2px,color:#fff
    classDef db fill:#006064,stroke:#4dd0e1,stroke-width:2px,color:#fff
    classDef kitchen fill:#1b5e20,stroke:#81c784,stroke-width:2px,color:#fff

    subgraph Customer ["1. Customer Point"]
        direction TB
        Voice(["Voice Interaction"])
        Tablet(["Table Tablet UI"])
    end

    subgraph Robot ["2. AI Service Robot (Jetson Orin)"]
        direction TB
        Agent["Intelligence & Agent Core<br>(LLM Orchestrator)"]
        Nav["Navigation & Task Control<br>(ROS 2 / Nav2)"]
    end

    subgraph Central ["3. Central Management System"]
        direction TB
        Dispatcher["Fleet & Order Dispatcher<br>(Rule-based Server)"]
        Database[("Central Database:<br>Menu, Orders, Shared Chat History")]
    end

    subgraph Kitchen ["4. Back-of-House"]
        KDS["Kitchen Display System"]
    end

    %% Customer <--> Robot Connections
    Voice <-->|"Voice In & TTS Out"| Agent
    Tablet <-->|"WebSocket: UI Events"| Agent

    %% Robot Internal Connections
    Agent -.->|"Transfer table_id"| Nav

    %% Robot <--> Central Connections
    Agent == "1. Write Order<br>2. Sync Shared Memory by Table ID" ==> Database
    Nav -. "Publish Telemetry / Heartbeat<br>(State: idle, location: kitchen)" .-> Dispatcher
    
    %% Server commands LLM
    Dispatcher -. "Assign Nav Mission<br>(Target: Table 5)" .-> Agent

    %% Central <--> Kitchen Connections
    Database <-->|"Read / Update Order Status"| KDS
    KDS -- "Chef completes order" --> Dispatcher

    %% Assign Classes
    class Voice,Tablet customer;
    class Agent,Nav robot;
    class Dispatcher central;
    class Database db;
    class KDS kitchen;
```

---

## 2. AI Brain: Orchestration & Layers
A focus on the 4-layer internal architecture of the AI service, from raw perception to action dispatching.

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart LR
    %% Styles
    classDef input fill:#0d47a1,stroke:#64b5f6,stroke-width:2px,color:#fff
    classDef process fill:#e65100,stroke:#ffb74d,stroke-width:2px,color:#fff
    classDef db fill:#006064,stroke:#4dd0e1,stroke-width:2px,color:#fff
    classDef tool fill:#bf360c,stroke:#ffcc80,stroke-width:2px,stroke-dasharray: 5 5,color:#fff
    classDef output fill:#1b5e20,stroke:#81c784,stroke-width:2px,color:#fff
    classDef ui fill:#4a148c,stroke:#ce93d8,stroke-width:2px,color:#fff

    %% 1. Input & Perception Layer
    subgraph Layer1 ["1. Perception Layer (Input)"]
        In(["Microphone"]) --> VAD["Silero VAD"]
        VAD -->|"Voice Detected"| STT["Pho-Whisper STT"]
        Touch(["iPad Touch Input"])
    end

    %% 2. Orchestration Layer
    subgraph Layer2 ["2. Orchestration Layer (Brain)"]
        STT -->|"Vietnamese Text"| Orchestrator{"LLM Orchestrator<br>+ System Prompt"}
        Touch -->|"UI Events / Item Selection"| Orchestrator
        Orchestrator <-->|"Loads/Saves via table_id"| ChatHist[("Shared Chat History DB")]
    end

    %% 3. Action & Tool Layer
    subgraph Layer3 ["3. Action Layer (Tools)"]
        Orchestrator -->|"JSON Tool Call"| Dispatcher{"Tool Dispatcher"}
        
        Dispatcher --> T_Search["Tool: search"]
        Dispatcher --> T_Order["Tool: place_order"]
        Dispatcher --> T_QR["Tool: qr_payment"]
        
        T_Search <-->|"Query"| RAG[("Vector DB: Sushi Menu")]
        T_Search -->|"Context Return"| Orchestrator
    end

    %% 4. Output & UI Layer
    subgraph Layer4 ["4. Output Layer (UI & Audio)"]
        Orchestrator -->|"Text Response"| TTS["TTS Engine"]
        TTS --> Out(["Robot Speaker"])
        
        Orchestrator -->|"State & Subtitles"| WS(("WebSocket Server"))
        T_Order -.->|"Order Status Updates"| WS
        T_QR -.->|"QR Code Payload"| WS
        
        WS <--> Monitor["iPad Monitor Display"]
    end

    %% Database & Hardware Exits
    T_Order == "DB Connection<br>(Write order_id & table_id)" ==> DB[("Central Restaurant DB")]
    Orchestrator -.->|"Semantic Intent<br>(e.g., 'Order Complete')"| Nav["ROS 2 Task Control<br>(Hardware Node)"]

    %% Assign Classes
    class In,VAD,Touch input;
    class STT,Orchestrator,TTS process;
    class RAG,ChatHist,DB db;
    class Dispatcher,T_Search,T_Order,T_QR tool;
    class Out,Nav output;
    class WS,Monitor ui;
```

---

## 3. Robotics: ROS 2 & Hardware Bridge
The integration between the AI Task Manager and the low-level hardware control using the Nav2 stack.

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart TD
    %% Styles
    classDef ai_node fill:#e65100,stroke:#ffb74d,stroke-width:2px,color:#fff
    classDef nav_node fill:#0d47a1,stroke:#64b5f6,stroke-width:2px,color:#fff
    classDef sensor_node fill:#006064,stroke:#4dd0e1,stroke-width:2px,color:#fff
    classDef hw_node fill:#1b5e20,stroke:#81c784,stroke-width:2px,color:#fff
    classDef topic color:#ffeb3b,font-size:12px,font-weight:bold

    subgraph AI_Bridge ["1. AI to ROS Bridge"]
        TaskManager["AI Task Manager Node<br>(Listens to LLM Intents)"]
    end

    subgraph Sensor_Drivers ["2. Perception Nodes (Jetson Orin)"]
        Lidar["LiDAR Driver Node"]
        DepthCam["RGB-D Camera Node"]
    end

    subgraph Core_Nav ["3. Navigation & Mapping (ROS 2 Nav2)"]
        SLAM["SLAM Toolbox / AMCL<br>(Localization)"]
        Costmap["Local & Global Costmaps"]
        Planner["Nav2 Planner & Controller<br>(Behavior Tree)"]
    end

    subgraph Low_Level ["4. Hardware Interface (Jetson to STM32)"]
        STM_Bridge["Micro-ROS / Serial Bridge Node"]
        PID_Kinematics["STM32:<br>Kinematics & PID Controller"]
    end

    %% AI to Nav2 Connection
    TaskManager == "Action: NavigateToPose<br>(Target Coordinates)" ===> Planner

    %% Sensor to Nav2 Connections
    Lidar -- "sensor_msgs/LaserScan<br>Topic: /scan" --> SLAM
    Lidar -- "Topic: /scan" --> Costmap
    DepthCam -- "sensor_msgs/PointCloud2<br>Topic: /camera/depth" --> Costmap

    %% SLAM to Nav2 Connection
    SLAM -- "nav_msgs/OccupancyGrid<br>Topic: /map" --> Planner
    SLAM -- "geometry_msgs/Pose<br>Topic: /amcl_pose" --> Planner

    %% Nav2 to Hardware Connection
    Planner == "geometry_msgs/Twist<br>Topic: /cmd_vel" ===> STM_Bridge

    %% Hardware to Jetson Feedback Loop
    STM_Bridge -- "nav_msgs/Odometry<br>Topic: /odom" --> SLAM
    STM_Bridge -- "Topic: /odom" --> Planner
    STM_Bridge <-->|"UART / Serial"| PID_Kinematics

    %% Assign Classes
    class TaskManager ai_node;
    class Planner,SLAM,Costmap nav_node;
    class Lidar,DepthCam sensor_node;
    class STM_Bridge,PID_Kinematics hw_node;
```
