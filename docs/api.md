# Spirit API Contract

## Auth
`POST /api/auth/register`  
body: `{"email":"you@example.com","password":"str0ng!"}`  
→ `{"access_token":"<jwt>","token_type":"bearer"}`

`POST /api/auth/login`  
form: `username=you@example.com&password=str0ng!`  
→ same token shape

## Goals
`POST /api/goals`  
auth header + JSON `{"text":"Write daily"}`  
→ `{"id":1,"user_id":1,"text":"Write daily","state":"active","created_at":"..."}`

`GET /api/goals/active`  
→ goal object or `null`

`PATCH /api/goals/{id}/complete` → `{"detail":"Goal marked completed"}`  
`PATCH /api/goals/{id}/abandon` → `{"detail":"Goal abandoned"}`

## Trajectory
`POST /api/trajectory/execute`  
`{"objective_text":"500 words","executed":true,"day":"2026-01-19"}`  
→ execution record

`GET /api/trajectory/history?limit=30`  
→ list of execution objects newest-first

## Strategic
`GET /api/strategic/status` → `{"unlocked":<bool>}`  
`POST /api/strategic/enter` → `{"detail":"Strategic mode unlocked ..."}` (or 403)

All timestamps UTC. All routes return 401 on bad/expired JWT.
