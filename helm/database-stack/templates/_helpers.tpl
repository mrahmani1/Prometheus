{{/*
Expand the name of the chart.
*/}}
{{- define "prometheus-database-stack.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "prometheus-database-stack.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "prometheus-database-stack.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "prometheus-database-stack.labels" -}}
helm.sh/chart: {{ include "prometheus-database-stack.chart" . }}
{{ include "prometheus-database-stack.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
environment: {{ .Values.environment }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "prometheus-database-stack.selectorLabels" -}}
app.kubernetes.io/name: {{ include "prometheus-database-stack.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "prometheus-database-stack.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "prometheus-database-stack.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Generate database connection strings
*/}}
{{- define "prometheus-database-stack.postgresqlConnectionString" -}}
postgresql://{{ .Values.postgresql.auth.username }}:{{ .Values.postgresql.auth.password }}@{{ include "prometheus-database-stack.fullname" . }}-postgresql:5432/{{ .Values.postgresql.auth.database }}
{{- end }}

{{- define "prometheus-database-stack.redisConnectionString" -}}
redis://{{ include "prometheus-database-stack.fullname" . }}-redis-master:6379
{{- end }}

{{- define "prometheus-database-stack.influxdbConnectionString" -}}
http://{{ include "prometheus-database-stack.fullname" . }}-influxdb:8086
{{- end }}

{{- define "prometheus-database-stack.neo4jConnectionString" -}}
bolt://{{ include "prometheus-database-stack.fullname" . }}-neo4j:7687
{{- end }}

{{- define "prometheus-database-stack.timescaledbConnectionString" -}}
postgresql://{{ .Values.timescaledb.auth.username }}:{{ .Values.timescaledb.auth.password }}@{{ include "prometheus-database-stack.fullname" . }}-timescaledb:5432/{{ .Values.timescaledb.auth.database }}
{{- end }}

{{/*
Generate storage class name
*/}}
{{- define "prometheus-database-stack.storageClassName" -}}
{{- if .Values.global.storageClass }}
{{- .Values.global.storageClass }}
{{- else }}
{{- "fast-ssd" }}
{{- end }}
{{- end }}