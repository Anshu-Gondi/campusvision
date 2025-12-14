export default function Dashboard() {
  return (
    <div className="content">
      <h1 className="title">Admin Dashboard</h1>
      <p className="subtitle">
        Organization & Branch level control panel
      </p>

      <div className="columns">
        <div className="column">
          <div className="box has-background-dark has-text-light">
            <p className="title is-4">🏫 Organizations</p>
            <p>Manage schools / colleges</p>
          </div>
        </div>

        <div className="column">
          <div className="box has-background-dark has-text-light">
            <p className="title is-4">🌍 Branches</p>
            <p>Manage locations & campuses</p>
          </div>
        </div>
      </div>
    </div>
  );
}
