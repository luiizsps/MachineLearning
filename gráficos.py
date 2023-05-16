# Gera gráficos
total_chamados = serie_temporal_chamados[:, 2]
total_vendas = serie_temporal_vendas[:, 2]
indices = indice[:, 2]
periodo = serie_temporal_vendas[:, [0, 1]]

# tarnsforma datas para formato string
datas_string = [f"{ano}-{mes:02d}" for ano, mes in periodo]

# reduz o numero de casas decimais da lista de indices
indicies_arredondados = [round(valor, 4) for valor in indices]

# transforma numero de vendas para plotar
total_vendas_str = []
for vendas in total_vendas:
    if(vendas >= 1000):
        total_vendas_str.append(f'{int(vendas/1000)}k')
    else:
        total_vendas_str.append(str(vendas))

total_vendas_nova_escala = total_vendas/20

# Dados de exemplo
data = {"x": datas_string, "y1": total_vendas_nova_escala, "y2": total_chamados, "y3": indices}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
sns.set_theme(style="white", context="talk")

sns.barplot(data=data, x="x", y="y2", color="lightblue", ax=ax1, label=barebone)
sns.lineplot(data=data, x="x", y="y1", marker='o', color="gray", ax=ax1, alpha=0.5, label="Parque")
ax1.axhline(0, color="k", clip_on=False)
ax1.set_title("Total de vendas e chamadas por mês", fontsize=13)
ax1.set_ylabel(" ")
ax1.set_ylim(0, np.max(total_vendas_nova_escala) * 1.5)

# Adicionando os valores no gráfico de barras do ax1
for i, bar in enumerate(ax1.patches):
    height = bar.get_height()
    ax1.annotate(str(data["y2"][i]), xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 1),
                 textcoords="offset points", ha='center', va='bottom', fontsize=8)

# Adicionando os valores no gráfico de linhas
i=0
for x_val, y_val in zip(data["x"], data["y1"]):
    ax1.annotate(str(total_vendas_str[i]), xy=(x_val, y_val), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8)
    i+=1
ax2.set_title("Índice de chamadas por total de vendas", fontsize=13)
ax2.axhline(0, color="k", clip_on=False)
sns.barplot(data=data, x="x", y="y3", color="lightblue", ax=ax2)
plt.xticks(rotation=45)

# Adicionando os valores no gráfico de barras do ax2
for i, bar in enumerate(ax2.patches):
    height = bar.get_height()
    ax2.annotate(str(indicies_arredondados[i]), xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 1),
                 textcoords="offset points", ha='center', va='bottom', fontsize=8)

sns.despine(bottom=True)
plt.setp(fig.axes, yticks=[])
plt.tight_layout(h_pad=2)

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, bbox_to_anchor = (1.05, 0.6), fontsize=12, title_fontsize='x-small')
# Exibindo a figura
plt.show()
