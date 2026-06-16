# Nebius Container Registry — Docker Push Skill
**נוצר:** יוני 2026 | **פרויקט:** MediSimplifier | **סטטוס:** ✅ עובד

---

## הבעיה שפתרנו

Docker Desktop לא עובד על Windows 1909 (ישן). רצינו לדחוף Docker image ל-Nebius Container Registry מ-Linux Build VM על ניביוס.

---

## תנאים מקדימים

- Build VM על ניביוס עם Docker מותקן (Ubuntu 24.04)
- Nebius CLI מותקן על ה-VM (`nebius --version`)
- Container Registry קיים בקונסולה

---

## תהליך מלא — שלב אחר שלב

### שלב 1 — צור Service Account בניביוס

1. לך ל: `https://console.nebius.com/tenant-{TENANT_ID}/iam/service-accounts`
2. לחץ **"Create entity" → "Service account"**
3. **Name:** `medisimplifier-sa`
4. **Project:** `default-project-eu-north1`
5. לחץ **"Create and continue"**
6. הוסף ל-group: **editors**
7. לחץ **Close**

**שמור את ה-Service Account ID:** `serviceaccount-e00vez9f3d2mv2p7ar`

---

### שלב 2 — צור Authorized Key (PEM format)

**ב-VM:**
```bash
openssl genrsa -out /tmp/private.pem 4096 && \
openssl rsa -in /tmp/private.pem -outform PEM -pubout -out /tmp/public.pem
```

**הורד את ה-public key למחשב:**
```powershell
# מהמחשב המקומי (PowerShell)
scp -i C:\Users\User\.ssh\nebius_vm ubuntu@{VM_IP}:/tmp/public.pem D:\Owner\Desktop\public.pem
```

**העלה לקונסולה:**
- לך ל-`medisimplifier-sa` → לשונית **"Authorized keys"**
- לחץ **"Upload authorized key"**
- בחר `public.pem` מה-Desktop
- שמור את ה-**Public Key ID:** `publickey-e00qxxdba17g1g1eag`

---

### שלב 3 — הגדר Nebius CLI על VM

```bash
nebius profile create
```

ענה על השאלות:
- **Profile name:** `default`
- **API endpoint:** `api.nebius.cloud`
- **Authorization type:** `service account`
- **Service account ID:** `serviceaccount-e00vez9f3d2mv2p7ar`
- **Public key ID:** `publickey-e00qxxdba17g1g1eag`
- **Path to PEM private key:** `/tmp/private.pem`
- **Tenant:** `tenant-e00ryr4vttfg4hrnxm | chambul-y83`
- **Project:** `project-e00g1ev2pr00wjxv40r6ga | default-project-eu-north1`

תוצאה: `profile "default" configured and activated`

---

### שלב 4 — הגדר Docker Credential Helper

```bash
# העתק config ל-root (נדרש ל-sudo)
sudo mkdir -p /root/.nebius && \
sudo cp ~/.nebius/config.yaml /root/.nebius/config.yaml && \
sudo nebius registry configure-helper
```

תוצאה: `creating new config /root/.docker/config.json`

---

### שלב 5 — Docker Login

```bash
# קבל IAM token ועשה login
nebius iam get-access-token | sudo docker login \
  cr.eu-north1.nebius.cloud \
  --username iam \
  --password-stdin
```

תוצאה: `Login Succeeded`

> ⚠️ **חשוב:** ה-token פג תוקף. יש לרענן לפני כל push session.

---

### שלב 6 — Build ו-Push

```bash
# Build
docker build \
  -t cr.eu-north1.nebius.cloud/{REGISTRY_ID}/medisimplifier:train-v1 \
  -f docker/Dockerfile.train .

# Push (עם sudo כי ה-login היה תחת sudo)
sudo docker push cr.eu-north1.nebius.cloud/{REGISTRY_ID}/medisimplifier:train-v1
```

---

## ערכים ספציפיים לפרויקט MediSimplifier

| פרמטר | ערך |
|--------|-----|
| Registry URL | `cr.eu-north1.nebius.cloud` |
| Registry ID | `e00p4ryvm6npw9w9pz` |
| Image מלא | `cr.eu-north1.nebius.cloud/e00p4ryvm6npw9w9pz/medisimplifier:train-v1` |
| Service Account | `serviceaccount-e00vez9f3d2mv2p7ar` |
| Public Key ID | `publickey-e00qxxdba17g1g1eag` |
| Tenant ID | `tenant-e00ryr4vttfg4hrnxm` |
| Project ID | `project-e00g1ev2pr00wjxv40r6ga` |
| Build VM IP | `89.169.122.1` (dynamic — יכול להשתנות) |
| SSH Key | `C:\Users\User\.ssh\nebius_vm` |

---

## בעיות שנתקלנו בהן ופתרונות

| בעיה | סיבה | פתרון |
|------|------|--------|
| `nebius registry configure-helper` — permission denied | חסר sudo | `sudo nebius registry configure-helper` |
| `sudo nebius registry configure-helper` — missing config | sudo משתמש ב-root שאין לו config | `sudo cp ~/.nebius/config.yaml /root/.nebius/` |
| `docker login` — incorrect auth format | username לא נכון | username חייב להיות `iam`, לא ה-key ID |
| `docker login` — failed to get profile | pipe עם sudo לא מעביר env vars | להשתמש ב-pipe: `nebius iam get-access-token \| sudo docker login ... --password-stdin` |
| `pip install easse` — `git checkout main` failed | branch נקרא `master` לא `main` | שנה ל-`@master` ב-requirements.txt |
| Access key (מחרוזת) לא עובד ל-CLI | CLI צריך Authorized key בפורמט PEM | צור RSA key pair עם openssl |

---

## פקודות מהירות לסשן עתידי

```bash
# 1. חבר ל-VM
ssh -i C:\Users\User\.ssh\nebius_vm ubuntu@{VM_IP}

# 2. git pull
cd medisimplifier-nebius && git pull

# 3. login (token מתחדש בכל פעם)
nebius iam get-access-token | sudo docker login \
  cr.eu-north1.nebius.cloud --username iam --password-stdin

# 4. build
docker build -t cr.eu-north1.nebius.cloud/e00p4ryvm6npw9w9pz/medisimplifier:train-v1 \
  -f docker/Dockerfile.train .

# 5. push
sudo docker push cr.eu-north1.nebius.cloud/e00p4ryvm6npw9w9pz/medisimplifier:train-v1

# 6. יציאה ועצירת VM
exit
# ואז בקונסולה: Compute → Instances → Stop
```

---

## הערות חשובות

- ה-IAM token פג תוקף כל ~12 שעות — יש לחדש לפני push
- ה-VM IP יכול להשתנות בכל הפעלה — בדוק בקונסולה
- ה-private key ב-`/tmp/private.pem` — אם VM נמחק, צריך לחזור על שלב 2
- לשמור את ה-private.pem במקום בטוח מחוץ ל-VM!
